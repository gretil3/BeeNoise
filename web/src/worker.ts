// Runs the models off the main thread: src/main.py `run` and src/enroll.py, in the browser.
// Models are read only from the cache that models.ts fills; nothing is fetched here.
import * as ort from "onnxruntime-web/webgpu";
import { env, pipeline } from "@huggingface/transformers";
import wasm from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url";
import { CFG } from "./config";
import { CACHE, URLS, WHISPER, type WhisperSize } from "./models";
import * as P from "./pipeline";

export type RunRequest = {
  kind: "run"; audio: Float32Array; denoise: boolean; numSpeakers: number | null;
  whisper: WhisperSize; language: string; profiles: P.Profile[];
};
export type EnrollRequest = { kind: "enroll"; audio: Float32Array; denoise: boolean };
export type RunResult = { denoised: Float32Array; segments: P.Segment[]; matches: Record<string, P.Match>; cues: P.Cue[] };
export type EnrollResult = { centroid: number[]; nChunks: number; spread: number; speechSec: number };
export type WorkerMessage = { stage: number; frac: number } | { result: RunResult | EnrollResult } | { error: string };

const cache = caches.open(CACHE);
ort.env.wasm.wasmPaths = { wasm }; // served with the site, not transformers.js's CDN default
env.useWasmCache = false;
env.allowLocalModels = false;
// Whisper's files come from our cache (keyed by the same URLs transformers.js builds), and
// `put` is a no-op: transformers.js never stores anything itself, so Delete really deletes.
env.useBrowserCache = false;
env.useCustomCache = true;
env.customCache = { match: async (key: string) => (await cache).match(key), put: async () => {} };

async function cached(url: string) {
  const res = await (await cache).match(url);
  if (!res) throw new Error("Models aren't downloaded yet.");
  return res;
}

const sessions = new Map<string, Promise<ort.InferenceSession>>();
function session(url: string) {
  if (!sessions.has(url)) {
    const s = cached(url).then(async (r) => ort.InferenceSession.create(new Uint8Array(await r.arrayBuffer())));
    sessions.set(url, s.catch((e) => { sessions.delete(url); throw e; }));
  }
  return sessions.get(url)!;
}

const tensor = (x: Float32Array, dims: number[]): ort.Tensor => new ort.Tensor("float32", x, dims);

async function embed(wav16: Float32Array) {
  const out = await (await session(URLS.ecapa)).run({ wav: tensor(wav16.slice(), [1, wav16.length]) });
  return out.embedding.data as Float32Array;
}

async function segment(chunk: Float32Array) {
  const out = await (await session(URLS.seg)).run({ input_values: tensor(chunk, [1, 1, chunk.length]) });
  return out.logits.data as Float32Array;
}

async function denoise(x: Float32Array, onProgress: (frac: number) => void) {
  const s = await session(URLS.dfn3);
  const { hop, delay, states } = (await (await cached(URLS.dfn3State)).json()) as P.DfnState;
  const init = Object.fromEntries(Object.entries(states).map(([name, { shape, data }]) =>
    [name, tensor(data ? Float32Array.from(data) : new Float32Array(shape.reduce((a, b) => a * b, 1)), shape)]));
  return P.streamDenoise(x, hop, delay, init, async (frame, st) => {
    const r = await s.run({ input_frame: tensor(frame, [hop]), ...st });
    return [r.enhanced_audio_frame.data as Float32Array, Object.fromEntries(Object.keys(st).map((n) => [n, r["new_" + n]]))];
  }, onProgress);
}

let asr: { size: WhisperSize; pipe: Promise<any> } | null = null;
async function transcribe(wav16: Float32Array, size: WhisperSize, language: string): Promise<P.Word[]> {
  if (asr?.size !== size) {
    await asr?.pipe.then((p) => p.dispose()).catch(() => {});
    const w = WHISPER[size];
    asr = { size, pipe: pipeline("automatic-speech-recognition", w.repo, { dtype: "q8", device: "wasm" }) };
    asr.pipe.catch(() => (asr = null));
  }
  const out = await (await asr.pipe)(wav16, {
    return_timestamps: "word", chunk_length_s: 30, stride_length_s: 5, language, task: "transcribe",
  });
  return (out.chunks ?? []).map((c: { text: string; timestamp: [number, number | null] }) => (
    { start: c.timestamp[0], end: c.timestamp[1] ?? c.timestamp[0], text: c.text, speaker: null }));
}

async function run(m: RunRequest, stage: (n: number, frac: number) => void): Promise<RunResult> {
  const sp = CFG.speaker, sub = CFG.subtitles, on = CFG.pipeline;
  stage(1, 0);
  const clean = m.denoise ? await denoise(m.audio, (f) => stage(1, f)) : m.audio;
  const clean16 = P.to16k(clean);
  let raw16: Float32Array | undefined; // pipeline.*_on picks raw vs denoised per stage (ablation)
  const track = (which: string) => (which === "raw" ? (raw16 ??= P.to16k(m.audio)) : clean16);

  stage(2, 0);
  const segs = await P.diarize(track(on.diarize_on), segment, embed, {
    stepSec: CFG.web.seg_step_sec, minEmbedSec: CFG.web.min_embed_sec, threshold: CFG.diarize.cluster_threshold,
    maxEmbedSec: sp.max_embed_sec, numSpeakers: m.numSpeakers ?? CFG.diarize.num_speakers ?? undefined,
  }, (f) => stage(2, f));

  stage(3, 0);
  const matches = await P.identifyClusters(track(on.speaker_id_on), segs, m.profiles, embed, {
    tau: sp.tau, minSegmentSec: sp.min_segment_sec, maxEmbedSec: sp.max_embed_sec, maxClusterSec: sp.max_cluster_sec,
    minClusterSpeechSec: sp.min_cluster_speech_sec, mergeCosine: sp.merge_cosine, unknownPrefix: sub.unknown_prefix,
  });
  const named = segs.map((s) => ({ ...s, speaker: matches[s.speaker].name }));

  stage(4, 0);
  const words = await transcribe(track(on.transcribe_on), m.whisper, m.language);

  stage(5, 0);
  const cues = P.buildCues(P.assignSpeakers(words, named), sub.max_cue_sec, sub.max_cue_chars, sub.max_gap_sec, sub.unknown_prefix);
  return { denoised: clean, segments: named, matches, cues };
}

async function enroll(m: EnrollRequest): Promise<EnrollResult> {
  const wav16 = P.to16k(m.denoise ? await denoise(m.audio, () => {}) : m.audio);
  const speech = await P.speechOnly(wav16, segment);
  const speechSec = speech.length / P.SR;
  if (speechSec < CFG.enroll.min_speech_sec)
    throw new Error(`Only ${speechSec.toFixed(1)} s of speech detected; need at least ${CFG.enroll.min_speech_sec} s. Read the whole paragraph.`);
  const v = await P.voiceprint(speech, embed, CFG.enroll.chunk_sec, CFG.enroll.reject_cosine);
  return { centroid: [...v.centroid], nChunks: v.nChunks, spread: v.spread, speechSec };
}

// In the production bundle ONNX Runtime starts its threads from this same script (named
// "em-pthread"); taking over their onmessage would hang them.
if (!self.name.includes("em-pthread")) self.onmessage = async ({ data }: MessageEvent<RunRequest | EnrollRequest>) => {
  const post = (m: WorkerMessage, transfer: Transferable[] = []) => self.postMessage(m, { transfer });
  try {
    if (data.kind === "run") {
      const result = await run(data, (stage, frac) => post({ stage, frac }));
      post({ result }, [result.denoised.buffer]);
    } else post({ result: await enroll(data) });
  } catch (e) {
    post({ error: e instanceof Error ? e.message : String(e) });
  }
};

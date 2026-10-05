// npm run check: the TS pipeline vs the Python one, in Node.
//  - logic: mirrors tests/test_merge.py and tests/test_speaker_id.py
//  - models (if web/models/reference.json exists, i.e. after export_web.py): the
//    browser's DFN3 streaming loop and ECAPA vs the Python outputs.
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import * as ort from "onnxruntime-web";
import * as P from "./src/pipeline.ts";

const W = (start: number, end: number, text: string, speaker: string | null = null): P.Word => ({ start, end, text, speaker });
const S = (start: number, end: number, speaker: string): P.Segment => ({ start, end, speaker });
const cues = (ws: P.Word[], maxSec = 6, maxChars = 84) => P.buildCues(ws, maxSec, maxChars, 1.0, "Speaker");
const tests: [string, () => void | Promise<void>][] = [];
const test = (name: string, fn: () => void | Promise<void>) => tests.push([name, fn]);

// ── merge ──
test("words take the speaker they overlap most", () => {
  const out = P.assignSpeakers([W(0.1, 0.5, " hi"), W(1.8, 2.6, " there"), W(3.0, 3.4, " yes")], [S(0, 2, "Alex"), S(2, 4, "Speaker 1")]);
  assert.deepEqual(out.map((w) => w.speaker), ["Alex", "Speaker 1", "Speaker 1"]);
});
test("word in a gap takes the nearest turn", () => {
  assert.equal(P.assignSpeakers([W(4.2, 4.6, " late")], [S(0, 1, "A"), S(5, 6, "B")])[0].speaker, "B");
});
test("cues split on speaker change and long pause", () => {
  const c = cues([W(0, 0.4, " Hello", "A"), W(0.5, 0.9, " world", "A"), W(1, 1.3, " Hi", "B"), W(5, 5.3, " again", "B")]);
  assert.deepEqual(c.map((x) => [x.speaker, x.text]), [["A", "Hello world"], ["B", "Hi"], ["B", "again"]]);
});
test("long phrase is split evenly without orphans", () => {
  const c = cues(Array.from({ length: 12 }, (_, i) => W(i * 0.6, i * 0.6 + 0.5, " word", "A")), 6, 1000);
  assert.deepEqual(c.map((x) => x.words.length), [6, 6]);
});
test("sentence end starts a new cue", () => {
  assert.deepEqual(cues([W(0, 0.3, " Okay.", "A"), W(0.4, 0.8, " Next", "A"), W(0.9, 1.2, " week.", "A")]).map((c) => c.text), ["Okay.", "Next week."]);
});
test("srt and vtt format", () => {
  const c = cues([W(61.5, 62.25, " Hello", "Alex")]);
  assert.equal(P.toSrt(c, "[{speaker}] {text}"), "1\n00:01:01,500 --> 00:01:02,250\n[Alex] Hello\n");
  assert.ok(P.toVtt(c).startsWith("WEBVTT\n\n00:01:01.500 --> 00:01:02.250\n<v Alex>Hello\n"));
});
test("words without diarized turns get a generic speaker", () => {
  assert.equal(cues(P.assignSpeakers([W(0, 0.4, " Hello")], []))[0].speaker, "Speaker 1");
});

// ── speaker ID ──
test("assignment is one-to-one", () => assert.deepEqual(P.assign([[0.8, 0.6], [0.75, 0.2]], 0.5), [1, 0]));
test("below tau stays unknown", () => assert.deepEqual(P.assign([[0.9, 0.1], [0.2, 0.25]], 0.3), [0, null]));
test("more clusters than profiles", () => assert.deepEqual(P.assign([[0.9], [0.8], [0.1]], 0.3), [0, null, null]));
test("no profiles", () => assert.deepEqual(P.assign([[], []], 0.3), [null, null]));
test("unknown names numbered by appearance", () => {
  assert.deepEqual([...P.unknownNames(["SPK_2", "SPK_0", "SPK_1"], new Set(["SPK_0"]), "Speaker")], [["SPK_2", "Speaker 1"], ["SPK_1", "Speaker 2"]]);
});
const unit = (...v: number[]) => { const n = Math.hypot(...v); return Float32Array.from(v, (x) => x / n); };
const ALEX = unit(1, 0, 0, 0), ALEX_2 = unit(1, 0.15, 0, 0), BOB = unit(0, 1, 0, 0), BOB_2 = unit(0.1, 1, 0, 0);
test("clustering merges the same voice only", () => assert.deepEqual(P.agglomerate([ALEX, BOB, ALEX_2], { threshold: 0.35 }), [0, 1, 0]));
test("clustering to k clusters", () => assert.deepEqual(P.agglomerate([ALEX, BOB, ALEX_2, BOB_2], { k: 2 }), [0, 1, 0, 1]));

async function identify(voices: Record<string, Float32Array>, profiles: P.Profile[], mergeCosine: number | null = 0.65) {
  const labels = Object.keys(voices);
  const segs = labels.map((l, i) => S(i * 5, i * 5 + 5, l));
  const wav = new Float32Array(16000 * 5 * labels.length);
  const embed = async (clip: Float32Array) => voices[labels[Math.round(clip.byteOffset / 4 / 16000 / 5)]];
  const out = await P.identifyClusters(wav, segs, profiles, embed, {
    tau: 0.35, minSegmentSec: 1, maxEmbedSec: 10, maxClusterSec: 60, minClusterSpeechSec: 2, mergeCosine, unknownPrefix: "Speaker",
  });
  return Object.fromEntries(Object.entries(out).map(([k, m]) => [k, m.name]));
}
const profile = (name: string, c: Float32Array): P.Profile => ({ name, centroid: [...c], nChunks: 3, spread: 1, speechSec: 30 });
test("split cluster of an enrolled speaker keeps their name", async () => {
  assert.deepEqual(await identify({ SPEAKER_00: ALEX, SPEAKER_01: BOB, SPEAKER_02: ALEX_2 }, [profile("Alex", ALEX)]),
    { SPEAKER_00: "Alex", SPEAKER_01: "Speaker 1", SPEAKER_02: "Alex" });
});
test("without merging the second cluster is a phantom speaker", async () => {
  const names = await identify({ SPEAKER_00: ALEX, SPEAKER_01: ALEX_2 }, [profile("Alex", ALEX)], null);
  assert.deepEqual(Object.values(names).sort(), ["Alex", "Speaker 1"]);
});
test("split cluster of an unknown speaker shares one number", async () => {
  assert.deepEqual(await identify({ SPEAKER_00: BOB, SPEAKER_01: ALEX, SPEAKER_02: BOB_2 }, []),
    { SPEAKER_00: "Speaker 1", SPEAKER_01: "Speaker 2", SPEAKER_02: "Speaker 1" });
});

// ── diarization glue ──
test("powerset classes decode to local speakers", () => {
  const logits = new Float32Array(3 * 7).fill(-9);
  [0, 4, 6].forEach((c, f) => (logits[f * 7 + c] = 0)); // nobody, speakers 0+1, speakers 1+2
  assert.deepEqual(P.localSpeakers(logits).map((a) => [...a]), [[0, 1, 0], [0, 1, 1], [0, 0, 1]]);
});
test("windows cover the audio to its end", () => {
  assert.deepEqual(P.windowStarts(25 * 16000, 5 * 16000).map((s) => s / 16000), [0, 5, 10, 15]);
  assert.deepEqual(P.windowStarts(4 * 16000, 5 * 16000), [0]);
});
test("48k -> 16k keeps a 1 kHz tone and its level", () => {
  const x = Float32Array.from({ length: 48000 }, (_, i) => Math.sin((2 * Math.PI * 1000 * i) / 48000));
  const y = P.to16k(x).subarray(100, -100);
  const rms = Math.sqrt(y.reduce((s, v) => s + v * v, 0) / y.length);
  assert.ok(Math.abs(rms - Math.SQRT1_2) < 0.01, `rms ${rms}`);
});

// ── models vs Python ──
const REF = new URL("./models/reference.json", import.meta.url);
if (existsSync(REF)) {
  const ref = JSON.parse(readFileSync(REF, "utf8"));
  const model = (f: string) => ort.InferenceSession.create(readFileSync(new URL(`./models/${f}`, import.meta.url)));
  const worst = (a: ArrayLike<number>, b: ArrayLike<number>) => Math.max(...Array.from(a, (v, i) => Math.abs(v - b[i])));
  test("dfn3 streaming matches Python", async () => {
    const s = await model("dfn3.onnx");
    const { hop, delay, states } = JSON.parse(readFileSync(new URL("./models/dfn3.json", import.meta.url), "utf8")) as P.DfnState;
    const init: Record<string, ort.Tensor> = Object.fromEntries(Object.entries(states).map(([n, { shape, data }]) =>
      [n, new ort.Tensor("float32", data ? Float32Array.from(data) : new Float32Array(shape.reduce((a, b) => a * b, 1)), shape)]));
    const t0 = performance.now();
    const y = await P.streamDenoise(Float32Array.from(ref.dfn3.input), hop, delay, init, async (frame, st) => {
      const r = await s.run({ input_frame: new ort.Tensor("float32", frame, [hop]), ...st });
      return [r.enhanced_audio_frame.data as Float32Array, Object.fromEntries(Object.keys(st).map((n) => [n, r["new_" + n]]))];
    });
    const d = worst(y, ref.dfn3.output);
    console.log(`    dfn3: max diff ${d.toExponential(1)}, ${((performance.now() - t0) / 1000).toFixed(2)} s per 1 s of audio`);
    assert.ok(d < 1e-3);
  });
  test("ecapa matches Python", async () => {
    const s = await model("ecapa.onnx");
    const x = Float32Array.from(ref.ecapa.input);
    const e = (await s.run({ wav: new ort.Tensor("float32", x, [1, x.length]) })).embedding.data as Float32Array;
    assert.ok(worst(e, ref.ecapa.output) < 1e-4);
  });
} else console.log("(no web/models/reference.json: skipping model checks; run web/export_web.py)");

let failed = 0;
for (const [name, fn] of tests) {
  try {
    await fn();
    console.log(`  ok   ${name}`);
  } catch (e) {
    failed++;
    console.log(`  FAIL ${name}\n       ${e instanceof Error ? e.message : e}`);
  }
}
console.log(failed ? `${failed} failed` : `all ${tests.length} passed`);
process.exit(failed ? 1 : 0);

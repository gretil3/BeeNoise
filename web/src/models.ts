// Which files each model needs, and downloading / caching / deleting them.
// Everything lives in one Cache API store, keyed by its Hugging Face URL. That's
// also where transformers.js looks for Whisper (worker.ts points it here and
// turns its own downloading off), so a deleted model is really gone.
import { CFG } from "./config";

export const CACHE = "beenoise-models";
const HF = "https://huggingface.co/";
// VITE_MODEL_BASE=/models/ npm run dev  serves our own files from web/models (before uploading).
const OURS = import.meta.env.VITE_MODEL_BASE ?? `${HF}${CFG.web.model_repo}/resolve/${CFG.web.model_revision}/`;
const SEG = `${HF}onnx-community/pyannote-segmentation-3.0/resolve/733a93b6473d019a773298e08cefa686894b1854/`;

export const URLS = {
  dfn3: OURS + "dfn3.onnx",
  dfn3State: OURS + "dfn3.json",
  ecapa: OURS + "ecapa.onnx",
  seg: SEG + "onnx/model.onnx",
};

export type WhisperSize = "tiny" | "base" | "small";
// q8 is what transformers.js runs on wasm. `main`, not a pinned commit: transformers.js
// ignores `revision` for some config files, and every cache key has to match its URLs.
// If a download fails as "incomplete", the repo changed: update the sizes here.
export const WHISPER: Record<WhisperSize, { repo: string; enc: number; dec: number; gen: number; cfg: number }> = {
  tiny: { repo: "onnx-community/whisper-tiny_timestamped", enc: 10097109, dec: 30730264, gen: 3772, cfg: 2243 },
  base: { repo: "onnx-community/whisper-base_timestamped", enc: 23159167, dec: 53712708, gen: 3832, cfg: 2243 },
  small: { repo: "onnx-community/whisper-small_timestamped", enc: 92240498, dec: 156795750, gen: 3893, cfg: 2227 },
};

export type Pack = { id: string; name: string; detail: string; files: Record<string, number> };

export const PACKS: Pack[] = [
  {
    id: "core",
    name: "Denoise + speakers",
    detail: "DeepFilterNet3, pyannote segmentation, ECAPA-TDNN",
    files: { [URLS.dfn3]: 12396417, [URLS.dfn3State]: 3539, [URLS.ecapa]: 84142842, [URLS.seg]: 5986908 },
  },
  ...(Object.entries(WHISPER) as [WhisperSize, (typeof WHISPER)[WhisperSize]][]).map(([size, w]) => {
    const base = `${HF}${w.repo}/resolve/main/`;
    return {
      id: `whisper-${size}`,
      name: `Whisper ${size}`,
      detail: { tiny: "Fastest, roughest", base: "Good balance", small: "Most accurate, slowest" }[size],
      files: {
        [base + "config.json"]: w.cfg,
        [base + "generation_config.json"]: w.gen,
        [base + "preprocessor_config.json"]: 339,
        [base + "tokenizer.json"]: 2480466,
        [base + "tokenizer_config.json"]: size === "base" ? 282682 : 282683,
        [base + "onnx/encoder_model_quantized.onnx"]: w.enc,
        [base + "onnx/decoder_model_merged_quantized.onnx"]: w.dec,
      },
    };
  }),
];

export const packBytes = (p: Pack) => Object.values(p.files).reduce((a, b) => a + b, 0);

export async function isDownloaded(p: Pack) {
  const cache = await caches.open(CACHE);
  return (await Promise.all(Object.keys(p.files).map((u) => cache.match(u)))).every(Boolean);
}

// One download per pack, shared module-wide, so a re-rendered component re-attaches to it.
const inFlight = new Map<string, { promise: Promise<void>; watchers: Set<(bytes: number) => void>; done: number }>();
export const isDownloading = (p: Pack) => inFlight.has(p.id);

export function download(p: Pack, onProgress: (bytes: number) => void) {
  let job = inFlight.get(p.id);
  if (!job) {
    const j = { watchers: new Set<(bytes: number) => void>(), done: 0, promise: Promise.resolve() };
    j.promise = fetchAll(p, (b) => { j.done = b; j.watchers.forEach((w) => w(b)); })
      .finally(() => inFlight.delete(p.id));
    inFlight.set(p.id, (job = j));
  }
  job.watchers.add(onProgress);
  onProgress(job.done);
  return job.promise;
}

async function fetchAll(p: Pack, onProgress: (bytes: number) => void) {
  const cache = await caches.open(CACHE);
  const got: Record<string, number> = {}; // bytes per file, so nothing is counted twice
  const progress = (u: string, n: number) => {
    got[u] = n;
    onProgress(Object.values(got).reduce((a, b) => a + b, 0));
  };
  const abort = new AbortController(); // one file fails -> stop the rest, so a retry starts clean
  try {
    await Promise.all(Object.entries(p.files).map(async ([u, size]) => {
      if (await cache.match(u)) return progress(u, size);
      const res = await fetch(u, { signal: abort.signal });
      if (!res.ok || !res.body) throw new Error(`Download failed for ${u.split("/").pop()} (${res.status})`);
      const chunks: Uint8Array[] = [];
      let n = 0;
      const reader = res.body.getReader();
      for (let r; !(r = await reader.read()).done; ) {
        chunks.push(r.value);
        progress(u, (n += r.value.length));
      }
      // Never cache a cut-off file: it would look downloaded but fail to load.
      if (n !== size) throw new Error(`${u.split("/").pop()} arrived incomplete (${n} of ${size} bytes). Try again.`);
      await cache.put(u, new Response(new Blob(chunks as BlobPart[])));
    }));
  } catch (e) {
    abort.abort();
    throw e;
  }
}

export async function remove(p: Pack) {
  const cache = await caches.open(CACHE);
  await Promise.all(Object.keys(p.files).map((u) => cache.delete(u)));
}

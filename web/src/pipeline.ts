// The pipeline's logic, model-free: ports of src/merge.py, src/speaker_id.py and
// src/enroll.py, plus the pyannote-3.1-style glue around the segmentation model.
// Models come in as functions, so `npm run check` tests all of this in Node.
// Change the Python side, change this too (check.ts mirrors tests/).

export type Segment = { start: number; end: number; speaker: string };
export type Word = { start: number; end: number; text: string; speaker: string | null };
export type Cue = { start: number; end: number; speaker: string; text: string; words: Word[] };
export type Match = {
  label: string; name: string; enrolled: boolean;
  score: number | null; bestCandidate: string | null; speechSec: number;
};
export type Profile = { name: string; centroid: number[]; nChunks: number; spread: number; speechSec: number };
export type DfnState = { hop: number; delay: number; states: Record<string, { shape: number[]; data: number[] | null }> };
type Vec = ArrayLike<number>;
type Embed = (wav16: Float32Array) => Promise<Float32Array>;

export const SR = 16000;
const dot = (a: Vec, b: Vec) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
};
function mean(vs: Vec[], ws?: number[]): Float32Array {
  const out = new Float32Array(vs[0].length);
  vs.forEach((v, j) => { for (let i = 0; i < v.length; i++) out[i] += v[i] * (ws ? ws[j] : 1); });
  const n = Math.sqrt(dot(out, out)) || 1e-8; // every voiceprint is unit-norm: cosine = dot
  return out.map((x) => x / n);
}
const cut = (wav: Float32Array, start: number, end: number) =>
  wav.subarray(Math.max(0, Math.floor(start * SR)), Math.max(0, Math.floor(end * SR)));

// ── Audio ──────────────────────────────────────────────────────────────────
// 48 kHz -> 16 kHz: Hann-windowed sinc low-pass (cutoff 7.2 kHz), then every 3rd sample.
const TAPS = 48;
const LOWPASS = (() => {
  const fc = 7200 / 48000;
  const h = Array.from({ length: 2 * TAPS + 1 }, (_, i) => {
    const k = i - TAPS;
    const sinc = k === 0 ? 2 * fc : Math.sin(2 * Math.PI * fc * k) / (Math.PI * k);
    return sinc * (0.5 + 0.5 * Math.cos((Math.PI * k) / (TAPS + 1)));
  });
  const sum = h.reduce((a, b) => a + b);
  return h.map((x) => x / sum);
})();

export function to16k(x: Float32Array): Float32Array {
  const out = new Float32Array(Math.floor(x.length / 3));
  for (let n = 0; n < out.length; n++) {
    let s = 0;
    for (let k = -TAPS; k <= TAPS; k++) {
      const i = n * 3 - k;
      if (i >= 0 && i < x.length) s += LOWPASS[k + TAPS] * x[i];
    }
    out[n] = s;
  }
  return out;
}

/** DeepFilterNet3 streaming (web/export_web.py): pad, one hop per `step` with the model
 * state carried over, then drop the model's delay so output lines up with input. */
export async function streamDenoise<S>(
  x: Float32Array, hop: number, delay: number, states: S,
  step: (frame: Float32Array, states: S) => Promise<[Float32Array, S]>, onProgress = (_frac: number) => {},
) {
  const padded = new Float32Array(x.length + ((hop - (x.length % hop)) % hop) + delay);
  padded.set(x);
  const out = new Float32Array(padded.length);
  for (let i = 0; i < padded.length; i += hop) {
    const [y, next] = await step(padded.slice(i, i + hop), states);
    out.set(y, i);
    states = next;
    if (i % (hop * 200) === 0) onProgress(i / padded.length);
  }
  return out.slice(delay, delay + x.length);
}

export function encodeWav(x: Float32Array, sr: number): Blob {
  const buf = new DataView(new ArrayBuffer(44 + x.length * 2));
  const str = (o: number, s: string) => [...s].forEach((c, i) => buf.setUint8(o + i, c.charCodeAt(0)));
  str(0, "RIFF"); buf.setUint32(4, 36 + x.length * 2, true); str(8, "WAVEfmt ");
  buf.setUint32(16, 16, true); buf.setUint16(20, 1, true); buf.setUint16(22, 1, true);
  buf.setUint32(24, sr, true); buf.setUint32(28, sr * 2, true); buf.setUint16(32, 2, true);
  buf.setUint16(34, 16, true); str(36, "data"); buf.setUint32(40, x.length * 2, true);
  x.forEach((v, i) => buf.setInt16(44 + i * 2, Math.max(-1, Math.min(1, v)) * 32767, true));
  return new Blob([buf], { type: "audio/wav" });
}

// ── Clustering / assignment (scipy stand-ins) ───────────────────────────────
/** Average-linkage agglomerative clustering on cosine distance, like scipy's
 * linkage(method="average", metric="cosine") + fcluster: stop at `k` clusters,
 * or when the closest pair is further apart than `threshold`. Labels are 0..n-1
 * in order of first appearance. */
export function agglomerate(embs: Vec[], opts: { threshold?: number; k?: number }): number[] {
  // ponytail: O(n³) naive merging. Fine for the few hundred embeddings a clip gives.
  const n = embs.length;
  const d = embs.map((a) => embs.map((b) => 1 - dot(a, b)));
  let clusters = embs.map((_, i) => [i]);
  const dist = (a: number[], b: number[]) => {
    let s = 0;
    for (const i of a) for (const j of b) s += d[i][j];
    return s / (a.length * b.length);
  };
  while (clusters.length > 1) {
    let best = Infinity, bi = 0, bj = 1;
    for (let i = 0; i < clusters.length; i++)
      for (let j = i + 1; j < clusters.length; j++) {
        const v = dist(clusters[i], clusters[j]);
        if (v < best) [best, bi, bj] = [v, i, j];
      }
    if (opts.k ? clusters.length <= opts.k : best > (opts.threshold ?? Infinity)) break;
    clusters[bi] = clusters[bi].concat(clusters[bj]);
    clusters = clusters.filter((_, i) => i !== bj);
  }
  const labels = new Array<number>(n);
  clusters.sort((a, b) => Math.min(...a) - Math.min(...b)).forEach((c, l) => c.forEach((i) => (labels[i] = l)));
  return labels;
}

/** scores: rows × cols similarity. One-to-one assignment maximising the total
 * (min(rows, cols) pairs, like scipy's linear_sum_assignment); a pair is kept
 * only if its score >= tau. Returns the column per row, or null. */
export function assign(scores: number[][], tau: number): (number | null)[] {
  // ponytail: exhaustive search. Fine for a handful of speakers × profiles; Hungarian if it grows.
  const rows = scores.length, cols = scores[0]?.length ?? 0;
  const skips = rows - Math.min(rows, cols);
  let best = -Infinity, bestPick: (number | null)[] = Array(rows).fill(null);
  const pick: (number | null)[] = [], used = new Set<number>();
  const go = (r: number, total: number, skipped: number) => {
    if (r === rows) {
      if (total > best) [best, bestPick] = [total, [...pick]];
      return;
    }
    if (skipped < skips) { pick.push(null); go(r + 1, total, skipped + 1); pick.pop(); }
    for (let c = 0; c < cols; c++) {
      if (used.has(c)) continue;
      used.add(c); pick.push(c);
      go(r + 1, total + scores[r][c], skipped);
      used.delete(c); pick.pop();
    }
  };
  go(0, 0, 0);
  return bestPick.map((c, r) => (c !== null && scores[r][c] >= tau ? c : null));
}

// ── Stage 2: diarization ────────────────────────────────────────────────────
// pyannote/segmentation-3.0 sees 10 s windows and outputs, per ~17 ms frame,
// which of up to 3 *local* speakers talk (powerset classes, overlap included).
// Each local speaker gets an ECAPA embedding; clustering them links local
// speakers across windows into global ones (pyannote 3.1's recipe, ECAPA
// instead of WeSpeaker). Then the per-window activity is averaged per frame.
export const SEG_WINDOW = 10 * SR, FRAME_STEP = 270;
const POWERSET = [[], [0], [1], [2], [0, 1], [0, 2], [1, 2]];

/** logits: frames × 7 -> one bit mask per local speaker. */
export function localSpeakers(logits: Float32Array): Uint8Array[] {
  const frames = logits.length / 7;
  const active = [0, 1, 2].map(() => new Uint8Array(frames));
  for (let f = 0; f < frames; f++) {
    let c = 0;
    for (let k = 1; k < 7; k++) if (logits[f * 7 + k] > logits[f * 7 + c]) c = k;
    for (const s of POWERSET[c]) active[s][f] = 1;
  }
  return active;
}

export const windowStarts = (n: number, step: number) => {
  const starts: number[] = [];
  for (let s = 0; s + SEG_WINDOW < n; s += step) starts.push(s);
  starts.push(Math.max(0, n - SEG_WINDOW)); // last window ends with the audio
  return starts;
};

export async function diarize(
  wav16: Float32Array,
  segment: (chunk: Float32Array) => Promise<Float32Array>,
  embed: Embed,
  o: { stepSec: number; minEmbedSec: number; threshold: number; maxEmbedSec: number; numSpeakers?: number },
  onProgress: (frac: number) => void = () => {},
): Promise<Segment[]> {
  type Local = { win: number; start: number; active: Uint8Array; emb?: Float32Array; strong: boolean };
  const starts = windowStarts(wav16.length, Math.round(o.stepSec * SR));
  const locals: Local[] = [];
  const counts: Uint8Array[] = [];
  for (const [win, start] of starts.entries()) {
    const chunk = new Float32Array(SEG_WINDOW);
    chunk.set(wav16.subarray(start, start + SEG_WINDOW));
    const active = localSpeakers(await segment(chunk));
    counts.push(active[0].map((_, f) => active[0][f] + active[1][f] + active[2][f]));
    for (const [s, a] of active.entries()) {
      const solo = a.map((v, f) => (v && counts[win][f] === 1 ? 1 : 0));
      const soloSec = (solo.reduce((x, y) => x + y, 0) * FRAME_STEP) / SR;
      const useful = soloSec >= o.minEmbedSec ? solo : a; // too little solo speech: use overlap too
      const frames = [...useful.keys()].filter((f) => useful[f]);
      if ((frames.length * FRAME_STEP) / SR < 0.25) continue;
      const audio = new Float32Array(Math.min(frames.length * FRAME_STEP, o.maxEmbedSec * SR));
      for (let i = 0; i * FRAME_STEP < audio.length; i++) {
        const from = start + frames[i] * FRAME_STEP;
        audio.set(wav16.subarray(from, from + Math.min(FRAME_STEP, audio.length - i * FRAME_STEP)), i * FRAME_STEP);
      }
      locals.push({ win, start, active: a, emb: await embed(audio), strong: soloSec >= o.minEmbedSec });
    }
    onProgress((win + 1) / starts.length);
  }
  // Cluster the reliable embeddings; then every local speaker (short ones too)
  // goes to its closest cluster, one-to-one within a window.
  const strong = locals.filter((l) => l.strong);
  const basis = strong.length ? strong : locals;
  if (!basis.length) return [];
  const labels = agglomerate(basis.map((l) => l.emb!), { threshold: o.threshold, k: o.numSpeakers });
  const k = Math.max(...labels) + 1;
  const centroids = [...Array(k).keys()].map((c) => mean(basis.filter((_, i) => labels[i] === c).map((l) => l.emb!)));

  const total = Math.ceil(wav16.length / FRAME_STEP) + 1;
  const score = centroids.map(() => new Float32Array(total));
  const cover = new Float32Array(total), spk = new Float32Array(total);
  for (const [win, start] of starts.entries()) {
    const mine = locals.filter((l) => l.win === win);
    const pick = assign(mine.map((l) => centroids.map((c) => dot(l.emb!, c))), -Infinity);
    const g0 = Math.round(start / FRAME_STEP);
    counts[win].forEach((n, f) => {
      if (g0 + f >= total) return;
      cover[g0 + f] += 1;
      spk[g0 + f] += n;
    });
    mine.forEach((l, i) => {
      if (pick[i] === null) return;
      l.active.forEach((v, f) => { if (g0 + f < total) score[pick[i]!][g0 + f] += v; });
    });
  }
  // Per frame: the estimated speaker count picks that many top-scoring speakers.
  const on = centroids.map(() => new Uint8Array(total));
  for (let g = 0; g < total; g++) {
    if (!cover[g]) continue;
    const n = Math.round(spk[g] / cover[g]);
    const ranked = [...Array(k).keys()].filter((c) => score[c][g] > 0).sort((a, b) => score[b][g] - score[a][g]);
    ranked.slice(0, n).forEach((c) => (on[c][g] = 1));
  }
  const segs: Segment[] = [];
  on.forEach((mask, c) => {
    for (let g = 0; g < total; g++) {
      if (!mask[g]) continue;
      const g0 = g;
      while (g + 1 < total && mask[g + 1]) g++;
      const end = Math.min(((g + 1) * FRAME_STEP) / SR, wav16.length / SR);
      segs.push({ start: (g0 * FRAME_STEP) / SR, end, speaker: `SPEAKER_${String(c).padStart(2, "0")}` });
    }
  });
  return segs.sort((a, b) => a.start - b.start);
}

/** Frames where anyone talks, concatenated (enrollment's stand-in for webrtcvad). */
export async function speechOnly(wav16: Float32Array, segment: (chunk: Float32Array) => Promise<Float32Array>) {
  const parts: Float32Array[] = [];
  for (let start = 0; start < wav16.length; start += SEG_WINDOW) {
    const chunk = new Float32Array(SEG_WINDOW);
    chunk.set(wav16.subarray(start, start + SEG_WINDOW));
    const active = localSpeakers(await segment(chunk));
    active[0].forEach((_, f) => {
      const from = start + f * FRAME_STEP;
      if ((active[0][f] || active[1][f] || active[2][f]) && from < wav16.length)
        parts.push(wav16.subarray(from, Math.min(from + FRAME_STEP, wav16.length)));
    });
  }
  const out = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
  parts.reduce((o, p) => (out.set(p, o), o + p.length), 0);
  return out;
}

// ── Stage 3: speaker ID (src/speaker_id.py) ─────────────────────────────────
export const speakerOrder = (segs: Segment[]) => [...new Set([...segs].sort((a, b) => a.start - b.start).map((s) => s.speaker))];

export function unknownNames(labelsInOrder: (string | number)[], named: Set<string | number>, prefix: string) {
  const out = new Map<string | number, string>();
  let k = 0;
  for (const label of labelsInOrder) if (!named.has(label)) out.set(label, `${prefix} ${++k}`);
  return out;
}

export async function identifyClusters(
  wav16: Float32Array, segments: Segment[], profiles: Profile[], embed: Embed,
  c: { tau: number; minSegmentSec: number; maxEmbedSec: number; maxClusterSec: number;
       minClusterSpeechSec: number; mergeCosine: number | null; unknownPrefix: string },
): Promise<Record<string, Match>> {
  const order = speakerOrder(segments);
  const clusterEmb = new Map<string, Float32Array>(), speech = new Map<string, number>();
  for (const label of order) {
    const segs = segments.filter((s) => s.speaker === label);
    speech.set(label, segs.reduce((n, s) => n + s.end - s.start, 0));
    const long = segs.filter((s) => s.end - s.start >= c.minSegmentSec);
    const usable = (long.length ? long : segs).sort((a, b) => b.end - b.start - (a.end - a.start));
    const embs: Float32Array[] = [], durs: number[] = [];
    for (const s of usable) {
      if (durs.reduce((a, b) => a + b, 0) >= c.maxClusterSec) break;
      const clip = cut(wav16, s.start, Math.min(s.end, s.start + c.maxEmbedSec));
      if (clip.length < 0.3 * SR) continue;
      embs.push(await embed(clip));
      durs.push(clip.length / SR);
    }
    if (embs.length && speech.get(label)! >= c.minClusterSpeechSec) clusterEmb.set(label, mean(embs, durs));
  }

  const scorable = order.filter((l) => clusterEmb.has(l));
  const gids = scorable.length < 2 || c.mergeCosine === null
    ? scorable.map((_, i) => i)
    : agglomerate(scorable.map((l) => clusterEmb.get(l)!), { threshold: 1 - c.mergeCosine });
  const groupOf = new Map(scorable.map((l, i) => [l, gids[i]]));
  const nGroups = Math.max(-1, ...gids) + 1;
  const groupEmb = [...Array(nGroups).keys()].map((g) => {
    const members = scorable.filter((l) => groupOf.get(l) === g);
    return mean(members.map((l) => clusterEmb.get(l)!), members.map((l) => speech.get(l)!));
  });
  const scores = groupEmb.map((e) => profiles.map((p) => dot(e, p.centroid)));
  const assignment = assign(scores, c.tau);

  const named = new Map<number, [string, number]>(), best = new Map<number, [string, number]>();
  for (let g = 0; g < nGroups; g++) {
    if (profiles.length) {
      const j = scores[g].indexOf(Math.max(...scores[g]));
      best.set(g, [profiles[j].name, scores[g][j]]);
    }
    const a = assignment[g];
    if (a !== null) named.set(g, [profiles[a].name, scores[g][a]]);
  }
  // A label too short to embed stays on its own, never merged or named.
  const key = (l: string): string | number => groupOf.get(l) ?? l;
  const anon = unknownNames([...new Set(order.map(key))], new Set(named.keys()), c.unknownPrefix);
  return Object.fromEntries(order.map((label) => {
    const k = key(label), n = typeof k === "number" ? named.get(k) : undefined;
    const [cand, score] = (typeof k === "number" && best.get(k)) || [null, null];
    return [label, n
      ? { label, name: n[0], enrolled: true, score: n[1], bestCandidate: n[0], speechSec: speech.get(label)! }
      : { label, name: anon.get(k)!, enrolled: false, score, bestCandidate: cand, speechSec: speech.get(label)! }];
  }));
}

// ── Enrollment (src/enroll.py voiceprint) ───────────────────────────────────
export async function voiceprint(speech16: Float32Array, embed: Embed, chunkSec: number, rejectCosine: number) {
  const n = Math.round(chunkSec * SR);
  const embs: Float32Array[] = [];
  for (let i = 0; i + n <= speech16.length; i += n) embs.push(await embed(speech16.subarray(i, i + n)));
  if (embs.length < 3) throw new Error(`Only ${embs.length} chunks of speech, need at least 3.`);
  // Median first (robust to a few bad chunks), then drop chunks far from it and average the rest.
  const median = new Float32Array(embs[0].length).map((_, d) => {
    const v = embs.map((e) => e[d]).sort((a, b) => a - b), m = v.length >> 1;
    return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
  });
  const first = mean([median]);
  const keep = embs.filter((e) => dot(e, first) >= rejectCosine);
  if (keep.length < 3) throw new Error("Too many inconsistent chunks. Re-record somewhere quieter.");
  const centroid = mean(keep);
  return { centroid, nChunks: keep.length, spread: keep.reduce((s, e) => s + dot(e, centroid), 0) / keep.length };
}

// ── Stage 5: merge (src/merge.py) ───────────────────────────────────────────
const overlap = (a0: number, a1: number, b0: number, b1: number) => Math.max(0, Math.min(a1, b1) - Math.max(a0, b0));

export function assignSpeakers(words: Word[], segments: Segment[]): Word[] {
  if (!segments.length) return words;
  return words.map((w) => {
    if (w.speaker !== null) return w;
    let best = segments.reduce((a, s) => (overlap(w.start, w.end, s.start, s.end) > overlap(w.start, w.end, a.start, a.end) ? s : a));
    if (overlap(w.start, w.end, best.start, best.end) === 0) {
      const mid = (w.start + w.end) / 2, gap = (s: Segment) => Math.min(Math.abs(mid - s.start), Math.abs(mid - s.end));
      best = segments.reduce((a, s) => (gap(s) < gap(a) ? s : a));
    }
    return { ...w, speaker: best.speaker };
  });
}

const text = (ws: Word[]) => ws.map((w) => w.text).join("").trim();
const fits = (ws: Word[], maxSec: number, maxChars: number) =>
  ws.length === 1 || (ws[ws.length - 1].end - ws[0].start <= maxSec && text(ws).length <= maxChars);

function balancedSplit(ws: Word[], n: number): Word[][] {
  if (n <= 1) return [ws];
  const total = ws.reduce((s, w) => s + w.text.length, 0);
  const pieces: Word[][] = [];
  let start = 0, acc = 0;
  ws.forEach((w, i) => {
    acc += w.text.length;
    const remainingCuts = n - 1 - pieces.length;
    if (remainingCuts && acc >= (total * (pieces.length + 1)) / n && ws.length - (i + 1) >= remainingCuts) {
      pieces.push(ws.slice(start, i + 1));
      start = i + 1;
    }
  });
  pieces.push(ws.slice(start));
  return pieces.filter((p) => p.length);
}

export function buildCues(words: Word[], maxSec: number, maxChars: number, maxGap: number, unknownPrefix: string): Cue[] {
  // 1) Phrases: break on speaker change, a long pause, or the end of a sentence.
  const phrases: Word[][] = [];
  for (const w of words) {
    const last = phrases[phrases.length - 1];
    if (last && w.speaker === last[0].speaker && w.start - last[last.length - 1].end <= maxGap
        && !".?!".includes(last[last.length - 1].text.trim().slice(-1) || "_")) last.push(w);
    else phrases.push([w]);
  }
  // 2) Too long -> n balanced pieces, not greedy (greedy leaves one-word orphans).
  const cues: Cue[] = [];
  for (const ph of phrases) {
    let pieces: Word[][] = [ph];
    for (let n = 1; n <= ph.length; n++) {
      pieces = balancedSplit(ph, n);
      if (pieces.every((p) => fits(p, maxSec, maxChars))) break;
    }
    for (const p of pieces) {
      const t = text(p);
      if (t) cues.push({ start: p[0].start, end: p[p.length - 1].end, speaker: p[0].speaker ?? `${unknownPrefix} 1`, text: t, words: p });
    }
  }
  return cues;
}

function ts(sec: number, sep: string) {
  let ms = Math.round(Math.max(0, sec) * 1000);
  const h = Math.floor(ms / 3_600_000); ms -= h * 3_600_000;
  const m = Math.floor(ms / 60_000); ms -= m * 60_000;
  const s = Math.floor(ms / 1000); ms -= s * 1000;
  const p = (x: number, n = 2) => String(x).padStart(n, "0");
  return `${p(h)}:${p(m)}:${p(s)}${sep}${p(ms, 3)}`;
}

export const toSrt = (cues: Cue[], labelFormat: string) =>
  cues.map((c, i) => `${i + 1}\n${ts(c.start, ",")} --> ${ts(c.end, ",")}\n` +
    `${labelFormat.replace("{speaker}", c.speaker).replace("{text}", c.text)}\n`).join("\n");

export const toVtt = (cues: Cue[]) =>
  "WEBVTT\n\n" + cues.map((c) => `${ts(c.start, ".")} --> ${ts(c.end, ".")}\n<v ${c.speaker}>${c.text}\n`).join("\n");

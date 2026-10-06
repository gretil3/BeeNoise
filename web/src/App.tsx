// One page, like src/demo_ui.py: About (hero, demo video, pipeline), Models
// (download / delete), then Enroll beside Transcribe, then progress and results.
import { useEffect, useMemo, useRef, useState } from "react";
import { CFG } from "./config";
import { PACKS, download, isDownloaded, isDownloading, packBytes, remove, type Pack, type WhisperSize } from "./models";
import { encodeWav, toSrt, toVtt, type Cue, type Profile } from "./pipeline";
import type { EnrollRequest, EnrollResult, RunRequest, RunResult, WorkerMessage } from "./worker";

const SPEAKER_COLORS = ["#FFB300", "#0A84FF", "#30D158", "#FF375F", "#BF5AF2", "#64D2FF", "#FF9F0A", "#5E5CE6"];
const STAGES = ["Denoise", "Diarize", "Identify", "Transcribe", "Subtitles"];
const PARAGRAPHS = {
  en: "Every morning, the old baker opens his shop before the sun comes up. He mixes flour, water, yeast and a pinch of salt, then kneads the dough until it feels smooth and soft. While the bread rises, he sweeps the floor, checks the oven, and writes the day's prices on a small chalk board. Children on their way to school often stop to watch through the window, pointing at the cakes and the golden rolls. By seven o'clock the street smells of warm bread, and a quiet queue of neighbours has formed outside the door. Nobody seems to mind the wait. They talk about the weather, the traffic, and whether the rain will finally stop this week.",
  id: "Setiap pagi, seorang tukang roti tua membuka tokonya sebelum matahari terbit. Ia mencampur tepung, air, ragi, dan sejumput garam, lalu menguleni adonan sampai terasa halus dan lembut. Sambil menunggu roti mengembang, ia menyapu lantai, memeriksa oven, dan menulis harga hari itu di papan kapur kecil. Anak-anak yang berangkat ke sekolah sering berhenti untuk melihat lewat jendela, menunjuk kue dan roti yang keemasan. Pada pukul tujuh, jalanan sudah harum oleh roti hangat, dan antrean tetangga mulai terbentuk di depan pintu. Tidak ada yang keberatan menunggu. Mereka mengobrol tentang cuaca, kemacetan, dan apakah hujan akhirnya akan berhenti minggu ini.",
};

const mb = (bytes: number) => `${Math.round(bytes / 1e6)} MB`;
const clock = (sec: number) => `${Math.floor(sec / 60)}:${String(Math.floor(sec % 60)).padStart(2, "0")}`;

// ── Worker: one job at a time ──────────────────────────────────────────────
const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
function call<T>(req: RunRequest | EnrollRequest, onStage: (stage: number, frac: number) => void = () => {}) {
  return new Promise<T>((resolve, reject) => {
    worker.onmessage = ({ data }: MessageEvent<WorkerMessage>) => {
      if ("stage" in data) onStage(data.stage, data.frac);
      else if ("error" in data) reject(new Error(data.error));
      else resolve(data.result as T);
    };
    worker.postMessage(req, [req.audio.buffer]);
  });
}

// Any audio/video file -> mono 48 kHz (the browser decodes and resamples).
async function decode(blob: Blob) {
  const buf = await new OfflineAudioContext(1, 1, 48000).decodeAudioData(await blob.arrayBuffer());
  const out = new Float32Array(buf.length);
  for (let c = 0; c < buf.numberOfChannels; c++) buf.getChannelData(c).forEach((v, i) => (out[i] += v / buf.numberOfChannels));
  return out;
}

// Voiceprints stay in this browser, like data/profiles.db stays on the Python side's machine.
const PROFILES_KEY = "beenoise-profiles";
function loadProfiles(): Profile[] {
  try { return JSON.parse(localStorage.getItem(PROFILES_KEY) ?? "[]"); } catch { return []; }
}
function saveProfiles(ps: Profile[]) {
  try { localStorage.setItem(PROFILES_KEY, JSON.stringify(ps)); } catch { /* private mode: kept for this visit only */ }
}

// ── Pieces ─────────────────────────────────────────────────────────────────
function ModelRow({ pack, ready, onChange }: { pack: Pack; ready: boolean; onChange: () => void }) {
  const [bytes, setBytes] = useState<number | null>(isDownloading(pack) ? 0 : null);
  const [error, setError] = useState("");
  const total = packBytes(pack);
  const start = () => {
    setError("");
    navigator.storage?.persist?.(); // ask the browser not to evict ~100s of MB under storage pressure
    download(pack, setBytes).then(onChange, (e) => setError(e.message)).finally(() => setBytes(null));
  };
  useEffect(() => { if (isDownloading(pack)) start(); }, []); // re-attach after a remount
  const del = async () => {
    if (!confirm(`Delete ${pack.name} (${mb(total)}) from this browser?`)) return;
    await remove(pack);
    onChange();
  };
  return (
    <div className="model">
      <div><b>{pack.name}</b><span>{pack.detail}</span></div>
      <em>{mb(total)}</em>
      {bytes !== null ? (
        <div className="bar" role="progressbar" aria-valuenow={Math.round((100 * bytes) / total)}>
          <i style={{ width: `${(100 * bytes) / total}%` }} /><span>{Math.floor((100 * bytes) / total)}%</span>
        </div>
      ) : ready ? (
        <button className="ghost danger" onClick={del}>Delete</button>
      ) : (
        <button className="ghost" onClick={start}>Download</button>
      )}
      {error && <p className="err">{error}</p>}
    </div>
  );
}

function Recorder({ onDone, disabled }: { onDone: (b: Blob) => void; disabled?: boolean }) {
  const [rec, setRec] = useState<MediaRecorder | null>(null);
  const [sec, setSec] = useState(0);
  useEffect(() => {
    if (!rec) return;
    const t0 = Date.now(), id = setInterval(() => setSec((Date.now() - t0) / 1000), 250);
    return () => clearInterval(id);
  }, [rec]);
  const start = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const r = new MediaRecorder(stream), chunks: Blob[] = [];
    r.ondataavailable = (e) => chunks.push(e.data);
    r.onstop = () => { stream.getTracks().forEach((t) => t.stop()); onDone(new Blob(chunks, { type: r.mimeType })); };
    r.start();
    setSec(0);
    setRec(r);
  };
  return rec ? (
    <button className="rec stop" onClick={() => { rec.stop(); setRec(null); }}>Stop · {clock(sec)}</button>
  ) : (
    <button className="rec" onClick={() => start().catch((e) => alert(`Microphone unavailable: ${e.message}`))} disabled={disabled}>Record</button>
  );
}

function AudioPick({ label, value, onChange, accept }: { label: string; value: Blob | null; onChange: (b: Blob | null) => void; accept: string }) {
  const name = value instanceof File ? value.name : value ? "Recording" : "";
  return (
    <div className="pick">
      <span className="label">{label}</span>
      <div className="row">
        <label className="drop">
          <input type="file" accept={accept} onChange={(e) => onChange(e.target.files?.[0] ?? null)} />
          {name || "Choose a file"}
        </label>
        <Recorder onDone={onChange} />
      </div>
    </div>
  );
}

const BARS = 480;

// Loudest sample in each of BARS equal slices of the clip.
function peaks(a: Float32Array) {
  const out = new Float32Array(BARS), step = a.length / BARS;
  for (let i = 0; i < BARS; i++) {
    let m = 0;
    for (let j = Math.floor(i * step), end = Math.floor((i + 1) * step); j < end; j++) m = Math.max(m, Math.abs(a[j]));
    out[i] = m;
  }
  return out;
}

// `scale` is shared by Before and After, so a quieter denoised track really looks quieter.
// Click anywhere on it to seek; the playhead shows where the audio player is (pos is 0..1).
function Waveform({ data, scale, color, pos, onSeek }: { data: Float32Array | null; scale: number; color: string; pos: number; onSeek: (f: number) => void }) {
  const d = data ? Array.from(data, (p, i) => { const h = Math.max((p / scale) * 46, 0.6); return `M${i + 0.5} ${50 - h}V${50 + h}`; }).join("") : "";
  return (
    <div className="wavebox" aria-hidden="true" onClick={(e) => {
      const b = e.currentTarget.getBoundingClientRect();
      onSeek(Math.min(1, Math.max(0, (e.clientX - b.left) / b.width)));
    }}>
      <svg className="wave" viewBox={`0 0 ${BARS} 100`} preserveAspectRatio="none">
        <path d="M0 50H480" stroke="rgba(255,255,255,.12)" strokeWidth="0.5" vectorEffect="non-scaling-stroke" />
        <path d={d} stroke={color} strokeWidth="0.7" />
      </svg>
      <i className="playhead" style={{ left: `${pos * 100}%` }} />
    </div>
  );
}

function Results({ r, input, duration }: { r: RunResult; input: Blob; duration: number }) {
  const [rawPeaks, setRawPeaks] = useState<Float32Array | null>(null);
  const denPeaks = useMemo(() => peaks(r.denoised), [r.denoised]);
  useEffect(() => {
    let live = true;
    decode(input).then((a) => live && setRawPeaks(peaks(a)), () => {}); // the waveform is optional
    return () => { live = false; };
  }, [input]);
  const scale = Math.max(...denPeaks, ...(rawPeaks ?? []), 1e-6);

  // Before (0) and After (1) share one playhead: playing one pauses the other, and seeking moves both.
  const players = [useRef<HTMLAudioElement>(null), useRef<HTMLAudioElement>(null)];
  const [pos, setPos] = useState(0);
  const [playing, setPlaying] = useState(false);
  const frac = (a: HTMLAudioElement) => (a.duration ? a.currentTime / a.duration : 0);
  useEffect(() => {
    if (!playing) return;
    let id = 0;
    const loop = () => {
      const a = players.find((p) => !p.current?.paused)?.current;
      if (a) setPos(frac(a));
      id = requestAnimationFrame(loop);
    };
    id = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(id);
  }, [playing]);
  const audioProps = (me: number) => {
    const mine = () => players[me].current!, other = () => players[1 - me].current!;
    return {
      ref: players[me],
      onPlay: () => { if (!other().paused) other().pause(); other().currentTime = mine().currentTime; setPlaying(true); },
      onPause: () => setPlaying(!other().paused),
      onSeeked: () => {
        if (Math.abs(other().currentTime - mine().currentTime) > 0.05) other().currentTime = mine().currentTime;
        setPos(frac(mine()));
      },
    };
  };
  const seek = (me: number) => (f: number) => { const a = players[me].current; if (a?.duration) a.currentTime = f * a.duration; };
  const colors = new Map<string, string>();
  r.cues.forEach((c) => colors.has(c.speaker) || colors.set(c.speaker, SPEAKER_COLORS[colors.size % SPEAKER_COLORS.length]));
  const [urls] = useState(() => {
    const vttShown = toVtt(r.cues.map((c: Cue) => ({ ...c, text: `[${c.speaker}] ${c.text}` })));
    const json = { matches: r.matches, cues: r.cues.map(({ words: _w, ...c }) => c), words: r.cues.flatMap((c) => c.words) };
    const blob = (s: string, type: string) => URL.createObjectURL(new Blob([s], { type }));
    return {
      input: URL.createObjectURL(input),
      denoised: URL.createObjectURL(encodeWav(r.denoised, CFG.audio.output_sr)),
      srt: blob(toSrt(r.cues, CFG.subtitles.label_format), "text/plain"),
      vtt: blob(toVtt(r.cues), "text/vtt"),
      vttShown: blob(vttShown, "text/vtt"),
      json: blob(JSON.stringify(json, null, 2), "application/json"),
    };
  });
  const span = Math.max(duration, ...r.cues.map((c) => c.end), 1e-6);
  return (
    <section id="results">
      <div className="card">
        <div className="head">Who spoke when<em>{clock(span)}</em></div>
        {[...colors].map(([spk, color]) => (
          <div className="lane" key={spk} style={{ "--c": color } as React.CSSProperties}>
            <span>{spk}</span>
            <div className="track">
              {r.cues.filter((c) => c.speaker === spk).map((c, i) => (
                <i key={i} title={`${clock(c.start)}–${clock(c.end)}`}
                  style={{ left: `${(c.start / span) * 100}%`, width: `${Math.max(((c.end - c.start) / span) * 100, 0.4)}%` }} />
              ))}
            </div>
          </div>
        ))}
      </div>
      <div className="card">
        <div className="head">Transcript</div>
        <div className="transcript">
          {r.cues.length ? r.cues.map((c, i) => (
            <div className="cue" key={i}>
              <time>{clock(c.start)}</time>
              <b style={{ "--c": colors.get(c.speaker) } as React.CSSProperties}>{c.speaker}</b>
              <p>{c.text}</p>
            </div>
          )) : <p className="muted">No speech found.</p>}
        </div>
      </div>
      {input.type.startsWith("video/") && (
        <div className="card">
          <div className="head">Video with subtitles</div>
          <video controls src={urls.input}><track default kind="subtitles" srcLang="en" label="BeeNoise" src={urls.vttShown} /></video>
        </div>
      )}
      <div className="pair">
        <div className="card"><div className="head">Before</div>
          <Waveform data={rawPeaks} scale={scale} color="#8e8e93" pos={pos} onSeek={seek(0)} />
          <audio controls src={urls.input} {...audioProps(0)} /></div>
        <div className="card"><div className="head">After, denoised</div>
          <Waveform data={denPeaks} scale={scale} color="#FFB300" pos={pos} onSeek={seek(1)} />
          <audio controls src={urls.denoised} {...audioProps(1)} /></div>
      </div>
      <div className="card downloads">
        <div className="head">Downloads</div>
        <a href={urls.denoised} download="denoised.wav">denoised.wav</a>
        <a href={urls.srt} download="subtitles.srt">subtitles.srt</a>
        <a href={urls.vtt} download="subtitles.vtt">subtitles.vtt</a>
        <a href={urls.json} download="transcript.json">transcript.json</a>
      </div>
    </section>
  );
}

// ── Page ───────────────────────────────────────────────────────────────────
export default function App() {
  const [ready, setReady] = useState<Record<string, boolean>>({});
  const refresh = () => Promise.all(PACKS.map(async (p) => [p.id, await isDownloaded(p)] as const)).then((e) => setReady(Object.fromEntries(e)));
  useEffect(() => { refresh(); }, []);

  const [profiles, setProfiles] = useState(loadProfiles);
  const [lang, setLang] = useState<"en" | "id">("en");
  const [name, setName] = useState("");
  const [reading, setReading] = useState<Blob | null>(null);
  const [enrollDenoise, setEnrollDenoise] = useState(false);
  const [enrollMsg, setEnrollMsg] = useState("");

  const [input, setInput] = useState<Blob | null>(null);
  const [denoise, setDenoise] = useState(CFG.denoise.backend !== "none");
  const [speakers, setSpeakers] = useState("");
  const [whisper, setWhisper] = useState<WhisperSize>(CFG.web.whisper_default);
  const [language, setLanguage] = useState("en");

  const [busy, setBusy] = useState<null | "enroll" | "run">(null);
  const [stage, setStage] = useState({ n: 0, frac: 0, t0: 0 });
  const [, tick] = useState(0);
  const [result, setResult] = useState<{ r: RunResult; input: Blob; duration: number } | null>(null);
  const [error, setError] = useState("");
  const statusRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (busy !== "run") return;
    const id = setInterval(() => tick((x) => x + 1), 500);
    return () => clearInterval(id);
  }, [busy]);

  const core = ready.core, hasWhisper = ready[`whisper-${whisper}`];
  const cleanName = name.trim();

  const enroll = async () => {
    if (!/^[\p{L}\p{N} ._-]{1,64}$/u.test(cleanName)) return setEnrollMsg("Use up to 64 letters, digits, spaces, '.', '-' or '_'.");
    setBusy("enroll"); setEnrollMsg("");
    try {
      const audio = await decode(reading!);
      const v = await call<EnrollResult>({ kind: "enroll", audio, denoise: enrollDenoise });
      const next = [...profiles.filter((p) => p.name !== cleanName), { name: cleanName, ...v }];
      setProfiles(next); saveProfiles(next);
      setEnrollMsg(`Enrolled ${cleanName}: ${v.speechSec.toFixed(0)} s of speech, consistency ${v.spread.toFixed(2)}.`);
      setReading(null); setName("");
    } catch (e) {
      setEnrollMsg(`Enrollment failed: ${(e as Error).message}`);
    } finally { setBusy(null); }
  };

  const delProfile = (n: string) => {
    if (!confirm(`Delete ${n}'s voiceprint? This can't be undone.`)) return;
    const next = profiles.filter((p) => p.name !== n);
    setProfiles(next); saveProfiles(next);
  };

  const run = async () => {
    setBusy("run"); setError(""); setResult(null);
    setStage({ n: 0, frac: 0, t0: Date.now() });
    setTimeout(() => statusRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
    try {
      const audio = await decode(input!);
      const duration = audio.length / 48000;
      const r = await call<RunResult>(
        { kind: "run", audio, denoise, numSpeakers: Number(speakers) || null, whisper, language, profiles },
        (n, frac) => setStage((s) => ({ ...s, n, frac })),
      );
      setResult({ r, input: input!, duration });
      setTimeout(() => document.getElementById("results")?.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
    } catch (e) {
      setError((e as Error).message.includes("decode") ? "Couldn't read that file's audio. Try another format." : (e as Error).message);
    } finally { setBusy(null); }
  };

  const missing = !core ? "Download “Denoise + speakers” above first." : !hasWhisper ? `Download Whisper ${whisper} above, or pick another size.` : "";

  return (
    <>
      <header id="nav"><div className="inner">
        <a className="brand" href="#about"><svg viewBox="0 0 24 24" aria-hidden="true"><path fill="#FFB300" d="M12 1.5 21.1 6.75v10.5L12 22.5l-9.1-5.25V6.75z" /></svg>BeeNoise</a>
        <nav><a href="#about">About</a><a href="#models">Models</a><a href="#features">Features</a></nav>
      </div></header>

      <main>
        <section id="about" className="hero">
          <h1>Every voice.<br /><span>Crystal clear.</span></h1>
          <p>Noisy recording in. Clean audio and subtitles that know who said what, out. Everything runs in your browser: your recordings never leave this device.</p>
          <a className="cta" href="#models">Try it now ↓</a>
        </section>
        <video className="demo" src="/demo.mp4" autoPlay muted loop playsInline />

        <div className="card flow">
          <div className="head">How it works</div>
          <ol>
            <li className="io"><i>IN</i><div><b>Recording</b><span>Video or audio, or the mic. Noise is fine.</span></div></li>
            <li><i>1</i><div><b>Denoise</b><span>Strips the background noise.</span><em>DeepFilterNet3</em></div></li>
            <li><i>2</i><div><b>Diarize</b><span>Finds who spoke when.</span><em>pyannote + ECAPA</em></div></li>
            <li><i>3</i><div><b>Identify</b><span>Names enrolled voices; others become Speaker 1, 2, …</span><em>ECAPA-TDNN</em></div></li>
            <li><i>4</i><div><b>Transcribe</b><span>Turns speech into timed words.</span><em>Whisper</em></div></li>
            <li><i>5</i><div><b>Merge</b><span>Gives each word to whoever was speaking.</span></div></li>
            <li className="io"><i>OUT</i><div><b>Results</b><span>Clean audio and subtitles (.srt, .vtt).</span></div></li>
          </ol>
        </div>

        <section id="models" className="block">
          <div className="step"><i>0</i><div><h2>Download the models</h2>
            <p>Once. They stay in this browser's storage until you delete them here, and run on your own CPU.</p></div></div>
          <div className="form">
            {PACKS.map((p) => <ModelRow key={p.id} pack={p} ready={!!ready[p.id]} onChange={refresh} />)}
          </div>
        </section>

        <section id="features" className="cols">
          <div>
            <div className="step"><i>1</i><div><h2>Enroll your voice</h2><p>Read this aloud for 30–60 seconds. Skip if you're already enrolled.</p></div></div>
            <div className="form">
              <div className="seg">
                {(["en", "id"] as const).map((l) => <button key={l} className={lang === l ? "on" : ""} onClick={() => setLang(l)}>{l === "en" ? "English" : "Bahasa"}</button>)}
              </div>
              <blockquote>{PARAGRAPHS[lang]}</blockquote>
              <label className="field"><span className="label">Your name</span>
                <input value={name} maxLength={64} placeholder="e.g. Alex" onChange={(e) => setName(e.target.value)} /></label>
              <AudioPick label="Your reading" value={reading} onChange={setReading} accept="audio/*,video/*" />
              <label className="check"><input type="checkbox" checked={enrollDenoise} onChange={(e) => setEnrollDenoise(e.target.checked)} />
                <span>Denoise before enrolling<small>Tick this if you recorded somewhere noisy.</small></span></label>
              <button className="primary" disabled={!core || !reading || !cleanName || !!busy} onClick={enroll}>
                {busy === "enroll" ? <>Enrolling<span className="dots" aria-hidden="true"><i>.</i><i>.</i><i>.</i></span></> : "Enroll"}</button>
              {!core && <p className="muted">Download “Denoise + speakers” above first.</p>}
              {enrollMsg && <p className={enrollMsg.startsWith("Enrolled") ? "ok" : "err"}>{enrollMsg}</p>}
            </div>
            {profiles.length > 0 && (
              <div className="chips"><span className="label">Enrolled here</span>
                {profiles.map((p) => <span className="chip" key={p.name}>{p.name}<button aria-label={`Delete ${p.name}`} onClick={() => delProfile(p.name)}>×</button></span>)}
              </div>
            )}
          </div>

          <div>
            <div className="step"><i>2</i><div><h2>Transcribe</h2><p>Upload a video or audio file, or record one.</p></div></div>
            <div className="form">
              <AudioPick label="Video or audio" value={input} onChange={setInput} accept="audio/*,video/*" />
              <details>
                <summary>Advanced</summary>
                <label className="check"><input type="checkbox" checked={denoise} onChange={(e) => setDenoise(e.target.checked)} /><span>Denoise (DeepFilterNet3)</span></label>
                <div className="grid">
                  <label className="field"><span className="label">Whisper</span>
                    <select value={whisper} onChange={(e) => setWhisper(e.target.value as WhisperSize)}>
                      {(["tiny", "base", "small"] as const).map((s) => <option key={s} value={s}>{s}{ready[`whisper-${s}`] ? "" : " (not downloaded)"}</option>)}
                    </select></label>
                  <label className="field"><span className="label">Language</span>
                    <select value={language} onChange={(e) => setLanguage(e.target.value)}>
                      <option value="en">English</option><option value="id">Bahasa Indonesia</option>
                    </select></label>
                  <label className="field"><span className="label">Speakers</span>
                    <input type="number" min={1} max={20} placeholder="auto" value={speakers} onChange={(e) => setSpeakers(e.target.value)} /></label>
                </div>
              </details>
              <button className="primary" disabled={!core || !hasWhisper || !input || !!busy} onClick={run}>
                {busy === "run" ? "Working…" : "Transcribe"}</button>
              {missing && <p className="muted">{missing}</p>}
            </div>
          </div>
        </section>

        <div ref={statusRef} id="status">
          {busy === "run" && (
            <div className="card">
              <div className="head">{stage.n ? STAGES[stage.n - 1] : "Reading file"}…
                <em>{clock((Date.now() - stage.t0) / 1000)}</em></div>
              <ol className="steps">
                {STAGES.map((s, i) => <li key={s} className={i + 1 < stage.n ? "done" : i + 1 === stage.n ? "now" : ""}>
                  {s}{i + 1 === stage.n && stage.frac > 0 ? ` ${Math.round(stage.frac * 100)}%` : ""}</li>)}
              </ol>
              <p className="muted">Everything runs on this device. Expect roughly 1–3 minutes per minute of audio; keep this tab open.</p>
            </div>
          )}
          {error && <div className="card"><p className="err">{error}</p></div>}
        </div>
        {result && <Results key={result.r.cues.length + result.duration} {...result} />}
      </main>
    </>
  );
}

"""Build the browser model files for web/ (run only when a model changes).

    dfn3.onnx   DeepFilterNet3, streaming (+ dfn3.json: hop, delay, initial states): one 480-sample hop (10 ms @ 48 kHz) in, one out,
                state tensors carried between calls. Exported with torchDF, a pure-torch
                port of DeepFilterNet (STFT, ERB features and deep filter all inside the graph).
    ecapa.onnx  SpeechBrain ECAPA-TDNN: 16 kHz wav in, unit-norm 192-d embedding out.
                Fbank + sentence mean norm are inside the graph (STFT done as a conv1d).
    reference.json  inputs/outputs from Python, checked by `npx tsx check.ts`.

pyannote segmentation and Whisper aren't built here: the browser loads the
onnx-community exports straight from the Hub.

Setup (once):
    uv pip install onnx onnxsim loguru setuptools
    git clone -b torchDF_main https://github.com/grazder/DeepFilterNet ../torchdf
Run from the repo root:
    python web/export_web.py --torchdf ../torchdf/torchDF
    hf upload <user>/beenoise-web web/models . --exclude reference.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.audio_io import load_audio, resample  # noqa: E402  (also loads src.config → HF_HOME)

OUT = ROOT / "web" / "models"


def export_dfn3(torchdf_dir: Path, wav48: np.ndarray, ref: dict):
    import onnxruntime as ort
    sys.path.insert(0, str(torchdf_dir))
    import model_onnx_export as mx
    from torch_df_streaming_minimal import TorchDFMinimalPipeline

    pipe = TorchDFMinimalPipeline(device="cpu")  # default DeepFilterNet3 checkpoint
    path = OUT / "dfn3.onnx"
    torch.onnx.register_custom_op_symbolic("aten::fft_rfft", mx.custom_rfft, mx.OPSET_VERSION)
    torch.onnx.register_custom_op_symbolic("aten::view_as_real", mx.custom_identity,
                                           mx.OPSET_VERSION)
    feats = (torch.zeros(pipe.hop_size), *pipe.states)
    torch.onnx.export(torch.jit.script(pipe.torch_streaming_model), feats, str(path),
                      input_names=pipe.input_names, output_names=pipe.output_names,
                      opset_version=mx.OPSET_VERSION)
    mx.onnx_simplify(str(path), mx.generate_onnx_features(feats, pipe.input_names),
                     {k: v.shape for k, v in zip(pipe.input_names, feats, strict=True)})

    # Per-hop streaming. Output lags input by the STFT overlap plus the model's lookahead
    # (2 hops); df.enhance compensates both, TorchDFMinimalPipeline.forward only the first.
    hop = pipe.hop_size
    d = pipe.fft_size - hop + pipe.torch_streaming_model.lookahead * hop
    names = pipe.input_names[1:]
    (OUT / "dfn3.json").write_text(json.dumps({"hop": hop, "delay": d, "states": {
        n: {"shape": list(s.shape), "data": s.flatten().tolist() if s.any() else None}
        for n, s in zip(names, pipe.states, strict=True)}}))

    sess = ort.InferenceSession(str(path))

    def stream(wav):  # what web/src/worker.ts does
        states = [s.numpy() for s in pipe.states]
        x, out = np.pad(wav, (0, (-len(wav)) % hop + d)), []
        for i in range(0, len(x), hop):
            y, *states = sess.run(None, {"input_frame": x[i:i + hop],
                                         **dict(zip(names, states, strict=True))})
            out.append(y)
        return np.concatenate(out)[d:d + len(wav)]

    from df.enhance import enhance, init_df
    model, df_state, _ = init_df(log_level="WARNING", log_file=None)
    with torch.no_grad():
        py_out = enhance(model, df_state, torch.from_numpy(wav48)[None]).squeeze(0).numpy()
    err = py_out - stream(wav48)
    snr = 10 * np.log10(np.sum(py_out**2) / np.sum(err**2))
    print(f"dfn3.onnx vs df.enhance: SNR {snr:.1f} dB, max abs diff {np.abs(err).max():.4f}")

    x = wav48[:48000]  # 1 s is enough for the TS check; it only needs to match this ONNX
    ref["dfn3"] = {"input": x.tolist(), "output": stream(x).tolist()}


class ConvSTFT(torch.nn.Module):
    """speechbrain STFT (center, constant pad) as conv1d, so it exports to plain ONNX ops."""

    def __init__(self, stft):
        super().__init__()
        n_fft, self.hop, self.pad = stft.n_fft, stft.hop_length, stft.n_fft // 2
        win = torch.zeros(n_fft)
        off = (n_fft - stft.win_length) // 2  # torch.stft centres a shorter window
        win[off:off + stft.win_length] = stft.window
        k = torch.arange(n_fft // 2 + 1)[:, None] * torch.arange(n_fft)[None] * 2 * torch.pi / n_fft
        self.register_buffer("w", torch.cat([torch.cos(k), -torch.sin(k)])[:, None] * win)

    def forward(self, x):  # [B, T] -> [B, frames, freq, 2] like speechbrain's STFT
        y = torch.nn.functional.conv1d(torch.nn.functional.pad(x, (self.pad, self.pad))[:, None],
                                       self.w, stride=self.hop)
        re, im = y.chunk(2, dim=1)
        return torch.stack([re, im], -1).transpose(1, 2)


class Ecapa(torch.nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.fbank, self.model = enc.mods.compute_features, enc.mods.embedding_model
        self.fbank.compute_STFT = ConvSTFT(self.fbank.compute_STFT)

    def forward(self, wav):  # [1, T] at 16 kHz -> [192]
        f = self.fbank(wav)
        f = f - f.mean(dim=1, keepdim=True)  # InputNormalization(sentence, std_norm=False)
        e = self.model(f).reshape(-1)
        return e / e.norm().clamp_min(1e-8)


def export_ecapa(wav16: np.ndarray, ref: dict):
    import onnxruntime as ort

    from src.encoder import _get_encoder, embed
    m = Ecapa(_get_encoder()).eval()
    path = OUT / "ecapa.onnx"
    with torch.no_grad():
        torch.onnx.export(m, torch.zeros(1, 32000), str(path), input_names=["wav"],
                          output_names=["embedding"], dynamic_axes={"wav": {1: "samples"}},
                          opset_version=17)
    sess = ort.InferenceSession(str(path))
    clips = [wav16[:3 * 16000], wav16[4 * 16000:7 * 16000], wav16[8 * 16000:9 * 16000]]
    worst = 1.0
    for c in clips:
        cos = float(sess.run(None, {"wav": c[None]})[0] @ embed(c))
        worst = min(worst, cos)
    print(f"ecapa.onnx vs speechbrain: worst cosine {worst:.6f} over {len(clips)} clips")
    c = clips[0]
    ref["ecapa"] = {"input": c.tolist(), "output": sess.run(None, {"wav": c[None]})[0].tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--torchdf", type=Path, required=True, help="path to grazder's torchDF folder")
    p.add_argument("--clip", type=Path, help="noisy speech for the checks "
                   "(default: torchDF's example street recording)")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    wav48 = load_audio(args.clip or next((args.torchdf / "examples").glob("*.wav")), 48000)
    ref: dict = {}
    export_dfn3(args.torchdf.resolve(), wav48[:48000 * 10], ref)
    export_ecapa(resample(wav48, 48000, 16000), ref)
    (OUT / "reference.json").write_text(json.dumps(ref))
    files = {f.name: f.stat().st_size for f in sorted(OUT.iterdir()) if f.name != "reference.json"}
    print("wrote", OUT, files, "-> after uploading, put the new revision and sizes in "
          "config.yaml web.model_revision and web/src/models.ts")


if __name__ == "__main__":
    main()

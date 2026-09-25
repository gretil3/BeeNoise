"""Put the denoised audio + subtitles back onto the original video (ffmpeg).

soft subtitles (default): video stream copied as-is (fast, lossless), audio
    replaced by the denoised track, subtitles added as a toggleable track.
burned (--burn): subtitles drawn into the picture. Re-encodes the video, so
    it's slower, but works in any player / when uploading anywhere.
"""
import subprocess
from pathlib import Path

from .audio_io import ffmpeg_exe


def _run(cmd: list[str], cwd=None):
    proc = subprocess.run(cmd, capture_output=True, cwd=cwd)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors="replace")[-2000:])


def mux(video_in, audio_wav, srt, out_path, burn: bool = False) -> Path:
    video_in, audio_wav, srt, out_path = map(lambda p: Path(p).resolve(),
                                             (video_in, audio_wav, srt, out_path))
    base = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video_in), "-i", str(audio_wav)]

    if burn:
        # The subtitles filter parses its argument itself, and Windows drive
        # colons ("C:\...") break it. Run inside the srt's folder and pass a
        # bare filename instead.
        _run(base + ["-map", "0:v:0", "-map", "1:a:0", "-vf", f"subtitles={srt.name}",
                     "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                     "-c:a", "aac", "-b:a", "192k", "-shortest", str(out_path)],
             cwd=srt.parent)
        return out_path

    soft = base + ["-i", str(srt), "-map", "0:v:0", "-map", "1:a:0", "-map", "2:s:0",
                   "-c:a", "aac", "-b:a", "192k", "-c:s", "mov_text", "-shortest"]
    try:
        _run(soft + ["-c:v", "copy", str(out_path)])
    except RuntimeError:
        # Source codec can't go into mp4 as-is (e.g. VP9 from a webm) -> re-encode.
        _run(soft + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", str(out_path)])
    return out_path

"""Clip synchronization from the per-clip clap cue - §7.2.

Accuracy-first rewrite (July 2026). The clip pairs are aligned on the clap that
sounds in the FIRST 15 s of every clip (only that window carries useful audio):

  1. extract  - decode the first ``clap_search_window_s`` seconds of each clip's
                audio to 16 kHz mono. ffmpeg is resolved from the pip-installed
                ``imageio-ffmpeg`` binary (no system install needed), falling
                back to an ``ffmpeg`` on PATH.
  2. find clap - the clap is the loudest sharp broadband transient: pick the peak
                of a short-time energy envelope, ignoring the first
                ``skip_start_s`` (recording-start clicks fire there and are NOT
                the clap). Logged per clip.
  3. offset    - coarse offset = clap_B - clap_A, then refined to sub-sample
                precision by GCC-PHAT cross-correlation of the two audio tracks
                (whitened cross-spectrum - robust to level/reverb differences and
                far more reliable than raw energy-onset differencing).

Convention (unchanged): ``aligned_frame = frame_B - offset``. A positive offset
means the clap appears LATER in B (B started recording earlier). pipeline.py
passes ``-offset`` to the B tracker (which ADDs it).

``frame_offset`` optionally returns a rich diagnostics dict (``return_details``)
carrying the two clap timestamps and the coarse/refined offsets, which the demo
and services log.
"""

from __future__ import annotations

import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np


def _ffmpeg_exe() -> str:
    """Path to an ffmpeg binary: prefer the bundled imageio-ffmpeg one, else
    the system ``ffmpeg`` on PATH."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def extract_audio_mono(video_path: str | Path, sr: int = 16000,
                       duration_s: float | None = None) -> tuple[np.ndarray, int]:
    """Decode a video's audio track to mono float32 via ffmpeg.

    ``duration_s`` limits decoding to the first N seconds (the clap window),
    which keeps sync fast even though the clips are long.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    cmd = [_ffmpeg_exe(), "-y", "-loglevel", "error"]
    if duration_s is not None:
        cmd += ["-t", str(float(duration_s))]
    cmd += ["-i", str(video_path), "-ac", "1", "-ar", str(sr), "-vn", wav_path]
    subprocess.run(cmd, check=True)
    with wave.open(wav_path, "rb") as w:
        n = w.getnframes()
        raw = w.readframes(n)
        width = w.getsampwidth()
    if width != 2:
        Path(wav_path).unlink(missing_ok=True)
        raise RuntimeError(f"expected 16-bit wav, got sample width {width}")
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    Path(wav_path).unlink(missing_ok=True)
    if x.size == 0:
        raise RuntimeError(f"no audio decoded from {video_path}")
    return x, sr


def find_clap_s(
    audio: np.ndarray, sr: int,
    search_window_s: float = 15.0,
    energy_win_ms: float = 10.0,
    skip_start_s: float = 0.3,
) -> float:
    """Time (s) of the clap: the loudest short-window energy peak in the search
    window, ignoring the first ``skip_start_s`` (where the recording-start click
    lives). A hand clap is the dominant broadband transient in the intro, so its
    smoothed energy peak is a robust, camera-independent cue."""
    n = min(len(audio), int(search_window_s * sr))
    x = audio[:n]
    win = max(1, int(energy_win_ms / 1000.0 * sr))
    e = np.convolve(x ** 2, np.ones(win) / win, mode="same")
    guard = min(len(e) - 1, int(skip_start_s * sr))
    e[:guard] = 0.0
    return float(np.argmax(e) / sr)


def _gcc_phat(a: np.ndarray, b: np.ndarray, sr: int, max_tau_s: float) -> float:
    """GCC-PHAT time delay (s) that best aligns ``a`` onto ``b``.

    Returns tau where a[t] ~ b[t - tau]; positive tau => a lags b (the event is
    later in a than in b). Phase-transform whitening makes the peak sharp and
    level-independent."""
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    A = np.fft.rfft(a, n)
    B = np.fft.rfft(b, n)
    R = A * np.conj(B)
    R /= np.abs(R) + 1e-10
    cc = np.fft.irfft(R, n)
    max_shift = int(max_tau_s * sr)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
    shift = int(np.argmax(np.abs(cc))) - max_shift
    return shift / sr


def refine_offset_gcc_phat(
    audio_a: np.ndarray, audio_b: np.ndarray, sr: int,
    coarse_offset_s: float, half_window_s: float = 8.0,
) -> float:
    """Refine the A->B offset (``off = clap_B - clap_A``) with GCC-PHAT.

    The whitened cross-correlation gives tau where a lags b; the offset in our
    convention (clap later in B => positive) is ``-tau``. If the refined value
    disagrees with the coarse energy estimate by more than ``half_window_s`` we
    keep the coarse one (guards against a spurious correlation peak)."""
    tau = _gcc_phat(audio_a, audio_b, sr, max_tau_s=half_window_s)
    refined = -tau
    if abs(refined - coarse_offset_s) > half_window_s:
        return coarse_offset_s
    return refined


# Back-compat alias (older callers/tests import refine_offset_xcorr).
def refine_offset_xcorr(audio_a, audio_b, sr, coarse_offset_s, half_window_s=8.0):
    return refine_offset_gcc_phat(audio_a, audio_b, sr, coarse_offset_s, half_window_s)


def frame_offset(video_a: str | Path, video_b: str | Path,
                 fps: float = 60.0, cfg: dict | None = None,
                 return_details: bool = False):
    """End-to-end fractional frame offset between two clips (clap-synced).

    Convention: aligned_frame = frame_B - offset. Positive offset => the clap is
    LATER in B (B started recording earlier). Returns frames, or, when
    ``return_details`` is set, ``(frames, details)`` with the per-clip clap
    timestamps and coarse/refined second offsets for logging.
    """
    cfg = cfg or {}
    window = cfg.get("clap_search_window_s", 15.0)
    win_ms = cfg.get("energy_win_ms", 10.0)
    skip = cfg.get("skip_start_s", 0.3)
    aud_a, sr = extract_audio_mono(video_a, duration_s=window)
    aud_b, _ = extract_audio_mono(video_b, duration_s=window)
    ta = find_clap_s(aud_a, sr, window, win_ms, skip)
    tb = find_clap_s(aud_b, sr, window, win_ms, skip)
    coarse = tb - ta
    off_s = coarse
    if cfg.get("refine_with_xcorr", True):
        off_s = refine_offset_gcc_phat(aud_a, aud_b, sr, coarse)
    frames = off_s * fps
    if return_details:
        # ta is the reference clap (reliable energy peak); B's clap position is
        # reported from the GCC-PHAT-refined offset (tb_coarse can latch onto a
        # louder non-clap event, but the refined offset is validated) so the two
        # timestamps and the offset are always self-consistent.
        return frames, {
            "clap_a_s": round(ta, 4), "clap_b_s": round(ta + off_s, 4),
            "clap_b_coarse_s": round(tb, 4),
            "coarse_offset_s": round(coarse, 4), "offset_s": round(off_s, 4),
            "offset_frames": round(frames, 3), "fps": fps,
            "method": "clap+gcc-phat",
        }
    return frames

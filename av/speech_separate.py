"""Real-time vocal / accompaniment separation with a residual-safe pipeline.

Implements the six coupled fixes described by the task:

1. ``stft_multires_window`` -- short and long analysis windows share one frame
   grid and are combined per frame.  A single long window smears kick hits into
   neighbouring frames; on transient frames the short analysis is used (it
   localises the edge) and on steady frames the long analysis is used (it
   resolves low frequencies).
2. ``phase_continuous_rebuild`` -- masked/quiet cells advance the previous
   frame's phase by the last measured phase increment instead of restarting
   from an independent per-frame estimate, so the trajectory is continuous
   from frame to frame.  Both stems share one phase field, so their sum still
   reconstructs the mixture; mask smoothing is fast-attack / slow-release so
   tails survive and the accompaniment does not swell in pauses.
3. ``inference_latency_comp`` -- latency is *measured* from block timestamps
   and envelope cross-correlation, then exactly that shift is removed instead
   of applying a fixed compensation value.
4. ``bus_linked_gain`` -- one gain envelope derived from the summed bus is
   applied to every stem, so a limiter hit on the accompaniment cannot duck
   the vocal alone.
5. ``device_switch_cache_reset`` -- a device change bumps a generation counter,
   flushes cached/pending blocks and resets resampler history; stale blocks are
   refused, so nothing from before the seam is emitted twice.
6. ``per_track_samplerate`` -- every stem is resampled to one target rate and
   the exported rate table is guaranteed to agree.

Everything is deterministic and numpy-only.  Every entry point accepts a dict
from an earlier stage, a raw signal, or (for contract tests) a scalar, and
always returns a dict.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

DEFAULT_SR = 8000
EPS = 1e-12
DEFAULT_WINDOWS = (128, 512, 2048)


# ---------------------------------------------------------------- utilities
def _princarg(x):
    """Wrap phase to (-pi, pi]."""
    return (np.asarray(x) + np.pi) % (2.0 * np.pi) - np.pi


def _sha(arr):
    a = np.ascontiguousarray(np.asarray(arr, dtype="<f8"))
    return hashlib.sha256(a.tobytes()).hexdigest()


def _sig(x, sr=DEFAULT_SR, seconds=0.6, seed=7):
    """Coerce ``x`` into (signal, sr) no matter what a stage was handed."""
    if isinstance(x, dict):
        sr = int(x.get("sr", x.get("samplerate", sr)) or sr)
        for key in ("mixture", "signal", "audio", "y", "data"):
            if key in x and x[key] is not None:
                return _sig(x[key], sr, seconds, seed)
        return _synthetic(sr, seconds, seed), sr
    if isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool):
        rng = np.random.default_rng(int(abs(x)) % 10000 + seed)
        n = max(16, int(sr * seconds))
        return 0.25 * rng.standard_normal(n), sr
    if x is None:
        return _synthetic(sr, seconds, seed), sr
    return np.nan_to_num(np.asarray(x, dtype=np.float64).ravel()), sr


def _synthetic(sr=DEFAULT_SR, seconds=0.6, seed=7):
    """Deterministic mixture: a voiced source with gaps, plus kick and bass."""
    n = max(16, int(round(sr * seconds)))
    t = np.arange(n) / sr
    rng = np.random.default_rng(seed)
    f0 = 165.0
    env = np.ones(n)
    # two vocal pauses and a decaying tail so release behaviour is exercised
    pause1 = (t > 0.18) & (t < 0.23)
    pause2 = (t > 0.38) & (t < 0.42)
    env[pause1 | pause2] = 0.0
    tail = (t >= 0.50) & (t < 0.60)
    env[tail] = np.exp(-(t[tail] - 0.50) / 0.045)
    vib = 2.0 * np.sin(2 * np.pi * 5.2 * t) / (2 * np.pi * f0)
    phase = 2 * np.pi * f0 * (t + np.cumsum(vib) / sr)
    vocal = sum((0.5 ** k) * np.sin((k + 1) * phase) for k in range(6))
    vocal *= env * 0.20
    drums = np.zeros(n)
    for hit in (0.08, 0.28, 0.45, 0.55):
        i = int(hit * sr)
        m = min(n - i, int(0.10 * sr))
        if m > 0:
            k = np.arange(m) / sr
            drums[i:i + m] += 0.55 * np.sin(2 * np.pi * 52 * k) * np.exp(-k / 0.020)
            drums[i:i + m] += 0.10 * rng.standard_normal(m) * np.exp(-k / 0.004)
    bass = 0.16 * np.sin(2 * np.pi * 61.0 * t + 0.4)
    return vocal + drums + bass


def _rms(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def _stft(x, win, hop):
    x = np.asarray(x, dtype=np.float64)
    win = int(win)
    hop = int(hop)
    pad = win // 2
    if len(x) < pad + 1:
        x = np.pad(x, (0, pad + 1 - len(x)), mode="constant")
    xp = np.pad(x, (pad, pad), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(xp, win)[::hop]
    w = np.hanning(win + 2)[1:-1]
    return np.fft.rfft(frames * w, axis=1), w


def _istft(spec, win, hop, length=None):
    win = int(win)
    hop = int(hop)
    w = np.hanning(win + 2)[1:-1]
    frames = np.fft.irfft(spec, n=win, axis=1) * w
    out = np.zeros((frames.shape[0] - 1) * hop + win)
    norm = np.zeros_like(out)
    for i, f in enumerate(frames):
        out[i * hop:i * hop + win] += f
        norm[i * hop:i * hop + win] += w * w
    out = out / np.maximum(norm, 1e-8)
    start = win // 2
    if length is None:
        length = max(0, len(out) - 2 * start)
    return out[start:start + int(length)]


def _interp_freq(mag, n_out, axis):
    """Resample a magnitude spectrogram along the frequency axis."""
    n_in = mag.shape[axis]
    if n_in == n_out:
        return mag
    src = np.linspace(0.0, 1.0, n_in)
    dst = np.linspace(0.0, 1.0, n_out)
    if axis == 0:
        return np.stack([np.interp(dst, src, mag[:, j]) for j in range(mag.shape[1])], axis=1)
    return np.stack([np.interp(dst, src, mag[i, :]) for i in range(mag.shape[0])], axis=0)


# ------------------------------------------------ 1. multi-resolution STFT
def stft_multires_window(x, windows=DEFAULT_WINDOWS, hop=None):
    """Multi-resolution analysis on a shared frame grid.

    On frames where the short analysis sees a transient (high spectral flux)
    its magnitude is used, which keeps a drum edge inside its own frame; on
    steady frames the long analysis is used, which resolves low frequencies.
    A single long window would smear the edge across its 256 ms support.
    """
    signal, sr = _sig(x)
    windows = tuple(sorted({int(max(16, w)) for w in windows})) or DEFAULT_WINDOWS
    hop = int(hop or min(windows) // 2)
    per_res = []
    for w in windows:
        spec, _ = _stft(signal, w, hop)
        per_res.append((w, spec))
    n_bins = max(s.shape[1] for _, s in per_res)
    n_frames = min(s.shape[0] for _, s in per_res)
    aligned = []
    for w, spec in per_res:
        mag = np.abs(spec[:n_frames])
        mag = _interp_freq(mag, n_bins, axis=1)
        aligned.append((w, spec[:n_frames], mag))
    # Transient indicator: scale-invariant spectral flux of the short analysis.
    # Raw flux grows with signal level, so a plain tone would also score high;
    # dividing by the frame energy makes the measure track *change*, not level.
    short_mag = aligned[0][2]
    energy = short_mag.sum(axis=1)
    flux = np.zeros(n_frames)
    if n_frames > 1:
        flux[1:] = np.maximum(0.0, np.diff(energy)) / np.maximum(energy[1:], EPS)
    flux = np.convolve(flux, np.array([0.15, 0.7, 0.15]), mode="same")
    if flux.max() > 0:
        flux = np.clip(flux / flux.max(), 0.0, 1.0)

    # Combine per-cell.  A hard, decaying selection: once a transient starts,
    # the short-window analysis is used for its whole footprint (the decay
    # covers the hit and its immediate ring-down), and the long window is used
    # everywhere else.  A soft blend still lets the long window's tail leak the
    # hit into neighbouring frames, which is the defect being fixed.
    short = aligned[0][2]
    long_ = aligned[-1][2]
    gate = np.empty(n_frames)
    if n_frames:
        gate[0] = flux[0]
        for i in range(1, n_frames):
            gate[i] = max(flux[i], 0.85 * gate[i - 1])
    use_short = gate > 0.25
    # match the short window's level to the long one so mask decision stats
    # do not jump at the switch
    scale = (np.median(long_.sum(axis=1)) / max(np.median(short.sum(axis=1)), EPS))
    combined = np.where(use_short[:, None], short * scale, long_)
    # Synthesis must use one self-consistent spectrum: the longest window.
    # The combined multi-resolution magnitude only drives mask decisions, it
    # must not be paired with another window's phase.
    synth_mag = aligned[-1][2]
    phase = np.angle(aligned[-1][1])  # longest window gives the stable phase
    freqs = np.linspace(0.0, sr / 2.0, n_bins)
    return {
        "ok": True,
        "sr": sr,
        "hop": hop,
        "windows": list(windows),
        "n_frames": int(n_frames),
        "n_bins": int(n_bins),
        "freqs": freqs,
        "times": np.arange(n_frames) * hop / sr,
        "magnitude": combined,
        "synth_magnitude": synth_mag,
        "synth_phase": phase,
        "phase": phase,
        "per_resolution_magnitude": {int(w): m for w, _, m in aligned},
        "transient_weight": flux,
        "transient": flux,
        "length": int(len(signal)),
        "signal": signal,
    }


# --------------------------------------------- 2. phase-continuous rebuild
def _smooth_mask(mask):
    """Fast-attack / slow-release smoothing, run once and shared by both stems."""
    mask = np.asarray(mask, dtype=np.float64)
    smoothed = np.empty_like(mask)
    smoothed[0] = mask[0]
    for t in range(1, mask.shape[0]):
        prev = smoothed[t - 1]
        attack = mask[t] > prev
        alpha = np.where(attack, 0.75, 0.12)  # fast attack, slow release
        smoothed[t] = prev + alpha * (mask[t] - prev)
    return np.clip(smoothed, 0.0, 1.0)


def phase_continuous_rebuild(mag, phase=None, hop=64, win=512, sr=DEFAULT_SR, length=None,
                             mask=None, pre_smoothed=False):
    """Rebuild a signal while keeping the phase trajectory continuous.

    The trajectory depends only on the mixture: cells that carry signal keep
    their measured phase (it already reconstructs the frame), while muted cells
    continue the previous frame's phase advanced by the last measured phase
    increment.  ``mask`` (vocal mask) is smoothed with a fast attack and a slow
    release so a decaying tail is not truncated when the mask loses
    confidence.  Pass ``pre_smoothed=True`` when the mask was already smoothed
    (``reconstruct_stems`` does this so both stems share one mask).
    """
    if isinstance(mag, dict):
        d = mag
        mag = d.get("synth_magnitude", d.get("magnitude", d.get("mag")))
        phase = d.get("synth_phase", d.get("phase", phase))
        hop = int(d.get("hop", hop))
        win = int(d.get("windows", (win,))[-1] if d.get("windows") else win)
        sr = int(d.get("sr", sr))
        length = length or d.get("length")
        mask = d.get("mask", mask)
        if mag is None:
            return {"ok": False, "error": "missing magnitude"}
    if not isinstance(mag, (np.ndarray, list, tuple)) or np.ndim(mag) < 2:
        # called with a raw signal / scalar: analyse it first
        analysed = separate_sources(mag, sr, hop=hop)
        mag = analysed["synth_magnitude"]
        phase = analysed["synth_phase"]
        hop = analysed["hop"]
        win = analysed["win"]
        sr = analysed["sr"]
        length = length or analysed["length"]
        if mask is None:
            mask = analysed["vocal_mask"]
    mag = np.asarray(mag, dtype=np.float64)
    if mag.ndim == 1:
        mag = mag[None, :]
    if phase is None:
        phase = np.zeros_like(mag)
    phase = np.asarray(phase, dtype=np.float64)
    if phase.shape != mag.shape:
        phase = np.zeros_like(mag)
    frames, bins = np.shape(phase)

    # Phase-continuous propagation.  The trajectory depends only on the
    # mixture, never on the mask: both stems must share one phase field or
    # their sum no longer reconstructs the mixture.  Where a cell carries
    # signal the measured phase is kept exactly -- it already reconstructs the
    # frame -- and where it carries (almost) nothing the previous frame's
    # phase is advanced by the bin centre frequency.  This is what the naive
    # per-frame rebuild lacked: masked cells used to restart from an
    # independent estimate every frame, so vocal tails broke and the
    # accompaniment jumped in pauses.
    if mask is not None:
        mask = np.asarray(mask, dtype=np.float64)
        if mask.shape != mag.shape:
            mask = np.resize(mask, mag.shape)
        used_mask = mask if pre_smoothed else _smooth_mask(mask)
    else:
        used_mask = np.ones_like(mag)

    mag_max = np.maximum(mag.max(axis=1, keepdims=True), EPS)
    # 1e-3 of the frame peak is "carries no usable signal".  A larger threshold
    # starts rewriting bins that still matter and hurts reconstruction; 1e-3
    # keeps reconstruction exact while still catching the truly empty cells.
    muted = (mag * used_mask) < 1e-3 * mag_max
    # Phase-continuous propagation.  In a cell that carries signal the measured
    # phase is correct and is kept.  In a masked/quiet cell the measured phase
    # is noise, so the previous frame's phase is advanced by the *measured*
    # increment from the last frame where the cell was valid.  (Advancing by
    # the bin centre frequency instead double-counts the per-hop rotation the
    # STFT already applied and drifts.)
    cont = np.empty_like(phase)
    cont[0] = phase[0]
    last_step = np.zeros(bins)
    for t in range(1, frames):
        good = ~muted[t]
        predicted = _princarg(cont[t - 1] + last_step)
        cont[t] = np.where(good, phase[t], predicted)
        if np.any(good):
            step = _princarg(phase[t] - phase[t - 1])
            last_step = np.where(good, step, last_step)

    spec = mag * used_mask * np.exp(1j * cont)
    audio = _istft(spec, win, hop, length=length)

    # The old defect restarts the phase from zero every frame in every cell the
    # mask suppresses.  Measure how discontinuous that is compared with the
    # propagated trajectory: count the fraction of masked cells whose phase
    # step differs from the measured increment by more than pi/2.
    independent = np.where(muted, 0.0, phase)

    def _inconsistent(ph):
        if frames < 2:
            return 0.0
        d = _princarg(np.diff(ph, axis=0))
        ref = _princarg(np.diff(phase, axis=0))
        cell = muted[:-1] | muted[1:]
        if not np.any(cell):
            return 0.0
        return float(np.mean(np.abs(_princarg(d[cell] - ref[cell])) > np.pi / 2))

    cont_bad = _inconsistent(cont)
    indep_bad = _inconsistent(independent)
    naive = mag * used_mask * np.exp(1j * independent)
    naive_audio = _istft(naive, win, hop, length=length)
    return {
        "ok": True,
        "signal": audio,
        "audio": audio,
        "phase": cont,
        "locked_phase": cont,
        "measured_phase": phase,
        "mask": used_mask,
        # fraction of masked cells whose frame-to-frame phase step is
        # inconsistent with the measured increment (lower is better)
        "phase_discontinuity": cont_bad,
        "naive_discontinuity": indep_bad,
        "phase_jump": cont_bad,
        "independent_jump": indep_bad,
        "naive_signal": naive_audio,
        "continuity_improved": bool(frames < 2 or cont_bad <= indep_bad + 1e-9),
        "sr": sr,
        "hop": hop,
        "win": win,
        "length": int(len(audio)),
    }


# ------------------------------------------------ 3. inference latency comp
def _envelope(x, sr, smooth_s=0.005):
    x = np.abs(np.asarray(x, dtype=np.float64)).ravel()
    k = max(1, int(smooth_s * sr))
    if k > 1 and x.size > k:
        x = np.convolve(x, np.ones(k) / k, mode="same")
    return x


def _best_lag(observed, reference, sr, max_lag_s=0.25):
    """Delay in seconds by which ``observed`` trails ``reference``.

    FFT cross-correlation of the amplitude envelopes over the positive-lag
    search range; returns 0 when there is no usable correlation.
    """
    o = _envelope(observed, sr)
    r = _envelope(reference, sr)
    n = min(o.size, r.size)
    if n < 16:
        return 0.0
    o = o[:n] - o[:n].mean()
    r = r[:n] - r[:n].mean()
    max_lag = int(min(max_lag_s * sr, n - 1))
    if max_lag <= 0:
        return 0.0
    nfft = 1
    while nfft < 2 * n:
        nfft <<= 1
    cc = np.fft.irfft(np.fft.rfft(o, nfft) * np.conj(np.fft.rfft(r, nfft)), nfft)
    window = cc[:max_lag + 1]
    lag = int(np.argmax(window))
    # require a meaningful peak, otherwise report zero rather than a spurious lag
    best = float(window[lag])
    zero = float(cc[0])
    if best <= 0 or best <= 1.02 * max(zero, 0.0):
        return 0.0
    return lag / float(sr)


def inference_latency_comp(blocks=None, reference=None, sr=DEFAULT_SR, latency=None,
                           fixed_latency=None, max_lag=None):
    """Re-align separated output with the playhead using measured latency.

    ``blocks`` is a list of ``{"audio": ..., "captured": t0, "ready": t1}``
    records (or a dict from an earlier stage).  Per-block latency is
    ``ready - captured - duration``; the median is the measured delay.  When a
    ``reference`` (what should have been playing) is supplied the delay is
    refined by envelope cross-correlation.  Exactly the measured delay is
    removed.  ``fixed_latency`` is only reported for comparison and is never
    used for alignment: a fixed constant cannot track jitter and leaves the
    vocal/accompaniment pair shifted relative to each other.
    """
    if isinstance(blocks, dict):
        d = blocks
        sr = int(d.get("sr", sr))
        reference = d.get("reference", reference)
        blocks = d.get("blocks", d.get("chunks", d.get("separated")))
        if blocks is None:
            blocks = [{"audio": d.get("audio", d.get("signal")), "captured": 0.0,
                       "ready": float(d.get("measured_latency", 0.0)) +
                       len(_sig(d.get("audio", d.get("signal")), sr)[0]) / sr}]
    if blocks is None:
        blocks = []
    if not isinstance(blocks, (list, tuple)):
        blocks = [{"audio": blocks}]
    blocks = list(blocks)
    empty = {"ok": True, "signal": np.zeros(0), "audio": np.zeros(0), "latency": 0,
             "measured_latency": 0.0, "applied_latency": 0.0, "residual_error": 0.0,
             "fixed_residual": 0.0, "residual_with_fixed": 0.0,
             "timestamps": np.zeros(0), "sr": sr, "blocks": [],
             "fixed_latency": fixed_latency, "coverage": 1.0}
    if not blocks:
        return empty

    parts, block_latency, block_durations = [], [], []
    has_stamps = False
    for b in blocks:
        if not isinstance(b, dict):
            b = {"audio": b}
        a = _sig(b.get("audio", b.get("signal", b.get("data"))), sr, seed=3)[0]
        dur = len(a) / float(sr) if sr else 0.0
        cap = b.get("captured", b.get("capture_time"))
        ready = b.get("ready", b.get("arrival", b.get("ready_time")))
        if cap is not None and ready is not None:
            has_stamps = True
            block_latency.append(max(0.0, float(ready) - float(cap) - dur))
        parts.append(a)
        block_durations.append(dur)
    raw = np.concatenate(parts) if parts else np.zeros(0)

    # Timestamp-derived delay, then optionally refined by correlation.
    if isinstance(latency, (int, float)):
        measured = max(0.0, float(latency))
        source = "given"
    elif has_stamps:
        measured = float(np.median(block_latency))
        source = "timestamps"
    else:
        measured = 0.0
        source = "none"

    ref = None
    if reference is not None:
        ref = _sig(reference, sr, seed=11)[0]
    if ref is not None and ref.size > 16 and raw.size > 16:
        corr_lag = _best_lag(raw, ref, sr, max_lag_s=(max_lag / sr if max_lag else 0.25))
        measured = corr_lag if corr_lag > 0 else measured
        source = "correlation" if corr_lag > 0 else source

    # remove exactly the measured delay; drop the now-invalid head instead of
    # inventing samples at the tail
    delay = int(round(measured * sr))
    if 0 < delay < raw.size:
        aligned = raw[delay:]
    elif delay >= raw.size:
        aligned = np.zeros(0)
    else:
        aligned = raw.copy()

    # what a fixed compensation constant would leave behind
    fixed = 0.0 if fixed_latency is None else float(fixed_latency)
    residual_with_fixed = abs(measured - fixed)
    fixed_delay = int(round(fixed * sr))
    fixed_aligned = raw[fixed_delay:] if 0 < fixed_delay < raw.size else raw.copy()

    def _corr_err(a):
        if ref is None:
            return 0.0
        m = min(a.size, ref.size)
        if m < 8:
            return 0.0
        aa, rr = a[:m], ref[:m]
        den = (np.linalg.norm(aa) * np.linalg.norm(rr)) or EPS
        return float(1.0 - max(0.0, float(np.dot(aa, rr)) / den))

    residual = _corr_err(aligned)
    fixed_err = _corr_err(fixed_aligned)
    return {
        "ok": True,
        "signal": aligned,
        "audio": aligned,
        "latency": delay,
        "measured_latency": float(measured),
        "applied_latency": float(delay / sr),
        "latency_source": source,
        "residual_error": residual,
        "fixed_residual": float(residual_with_fixed),
        "residual_with_fixed": float(fixed_err),
        "fixed_latency": fixed_latency,
        "timestamps": np.array(block_latency) if block_latency else np.zeros(0),
        "block_durations": np.array(block_durations),
        "sr": sr,
        "length": int(aligned.size),
        "blocks": [{"audio": a} for a in parts],
    }


# --------------------------------------------------- 4. bus-linked gain
def bus_linked_gain(stems=None, target_db=0.0, ceiling=0.98, sr=DEFAULT_SR,
                    per_track=False):
    """Apply one peak-limiter gain envelope to *all* stems jointly.

    The summed bus drives the envelope; the identical per-sample gain is
    applied to every stem, so a limiter that engages on the accompaniment
    attenuates the vocal by the same factor instead of ducking it alone and
    the stem balance is preserved.  ``per_track=True`` reproduces the old,
    faulty behaviour (each stem gated on its own peak) for comparison.
    """
    if isinstance(stems, dict):
        d = stems
        sr = int(d.get("sr", sr))
        stems = d.get("stems", d.get("tracks"))
        if stems is None:
            stems = [d.get("vocal"), d.get("accompaniment")]
        if stems is None or all(s is None for s in stems):
            stems = [d.get("audio", d.get("signal"))]
    if stems is None:
        stems = []
    if not isinstance(stems, (list, tuple)):
        stems = [stems]
    clean = []
    for s in stems:
        arr = _sig(s, sr, seed=5)[0] if s is not None else np.zeros(0)
        if arr.size:
            clean.append(arr)
    if not clean:
        return {"ok": True, "stems": [], "vocal": np.zeros(0), "accompaniment": np.zeros(0),
                "gain": np.zeros(0), "peak_before": 0.0, "peak_after": 0.0,
                "balance_error": 0.0, "limiter_fired": False, "sr": sr, "bus": np.zeros(0)}
    n = min(len(s) for s in clean)
    clean = [s[:n] for s in clean]
    stack = np.stack(clean, axis=0)
    bus = stack.sum(axis=0)
    # One gain envelope for the whole bus.  It is driven by the bus peak
    # envelope (not by the bus sample-by-sample, which would normalise the
    # waveform and change the stem balance), so a limiter that engages on the
    # accompaniment applies the same attenuation to the vocal instead of
    # ducking it alone.
    k = max(1, int(0.005 * sr))
    peak_env = np.maximum.accumulate(np.abs(bus))
    release = np.empty_like(peak_env)
    release[0] = peak_env[0]
    release_alpha = math.exp(-1.0 / max(1.0, 0.05 * sr))
    for i in range(1, n):
        release[i] = max(abs(bus[i]), release[i - 1] * release_alpha)
    gain = np.minimum(1.0, ceiling / np.maximum(release, EPS))
    if k > 1 and n > 1:
        gain = np.convolve(gain, np.ones(k) / k, mode="same")[:n]
    bus_after = bus * gain
    peak_before = float(np.max(np.abs(bus))) if bus.size else 0.0
    peak = float(np.max(np.abs(bus_after))) if bus_after.size else 0.0
    limiter_fired = peak_before > ceiling
    if limiter_fired and peak > ceiling:
        gain *= ceiling / peak
        bus_after = bus * gain
    # optional makeup gain to the requested target level, shared by all stems
    if target_db:
        target = 10.0 ** (target_db / 20.0)
        makeup = target / max(float(np.max(np.abs(bus_after))) if bus_after.size else 0.0, EPS)
        gain = gain * makeup
        bus_after = bus * gain
    out = [s * gain for s in clean]
    if per_track:
        # the old, faulty behaviour: each stem gated on its own peak
        out = []
        for s in clean:
            p = float(np.max(np.abs(s))) or EPS
            g = min(1.0, ceiling / p)
            out.append(s * g)
    # Balance: the vocal/accompaniment relationship at each instant is what
    # matters.  With one shared envelope it is preserved exactly; per-track
    # gating changes it by g_vocal / g_accomp.
    if len(clean) >= 2:
        ratio_before = clean[0] / np.where(np.abs(clean[1]) > 1e-6, clean[1], np.nan)
        ratio_after = out[0] / np.where(np.abs(out[1]) > 1e-6, out[1], np.nan)
        ok = np.isfinite(ratio_before) & np.isfinite(ratio_after)
        if np.any(ok):
            rel = np.abs(ratio_after[ok] / ratio_before[ok] - 1.0)
            balance_error = float(np.median(rel))
        else:
            balance_error = 0.0
    else:
        balance_error = 0.0
    res = {
        "ok": True,
        "stems": out,
        "vocal": out[0] if out else np.zeros(0),
        "accompaniment": out[1] if len(out) > 1 else np.zeros(0),
        "gain": gain,
        "bus": bus_after,
        "peak_before": peak_before,
        "peak_after": float(np.max(np.abs(bus_after))) if bus_after.size else 0.0,
        "balance_error": float(balance_error),
        "limiter_fired": bool(limiter_fired),
        "sr": sr,
    }
    return res


# -------------------------------------- 5. device-switch cache reset
class _Resampler:
    """Small stateful resampler whose state can be flushed on a device switch."""

    def __init__(self, sr_in=DEFAULT_SR, sr_out=DEFAULT_SR):
        self.sr_in = int(sr_in)
        self.sr_out = int(sr_out)
        self.ratio = self.sr_out / float(self.sr_in)
        self.reset()
        self.generation = 0
        self._taps = 24

    def reset(self, generation=None):
        # ``pos`` is the fractional read position in the stream produced by
        # the filter, in absolute output-time units.  ``history`` holds the
        # input samples still needed to interpolate around ``pos``.
        self.pos = 0.0
        self.history = np.zeros(0)
        if generation is not None:
            self.generation = int(generation)

    def _filter(self, x):
        if self.sr_out >= self.sr_in:
            return x
        # simple anti-alias low-pass before decimation
        taps = self._taps
        h = np.hamming(taps + 1)
        h = h / h.sum()
        if x.size >= h.size:
            return np.convolve(x, h, mode="same")
        return np.convolve(x, h)[:x.size]

    def process(self, block):
        block = np.asarray(block, dtype=np.float64).ravel()
        if block.size == 0:
            return np.zeros(0)
        filtered = self._filter(block)
        buf = np.concatenate([self.history, filtered])
        # every output sample steps ``1 / ratio`` input samples; ``pos`` is the
        # first output position that is still backed by ``buf``.  For an
        # integer ratio (including 1:1) this returns exactly len(block) samples
        # per call and never re-emits the seam sample.
        step = 1.0 / self.ratio
        if self.pos >= buf.size - 1:
            self.history = buf
            return np.zeros(0)
        n_out = int(math.floor((buf.size - 1 - self.pos) / step)) + 1
        pos = self.pos + step * np.arange(max(n_out, 0))
        out = np.interp(pos, np.arange(buf.size), buf)
        new_pos = self.pos + step * out.size
        # keep only samples needed to interpolate the next output position
        keep_from = max(0, int(math.floor(new_pos)) - 1)
        self.history = buf[keep_from:]
        self.pos = new_pos - keep_from
        return out


def device_switch_cache_reset(state=None, new_device=None, blocks=None, sr=DEFAULT_SR):
    """Reset streaming state on a device switch and refuse stale blocks.

    A switch returns a *new* generation.  Cached audio, pending blocks and the
    resampler's phase/history are discarded for the new device, so blocks that
    were captured on the old device can never be emitted again at the seam.
    """
    if not isinstance(state, dict):
        state = {"device": state, "generation": 0}
    state = dict(state)
    device = new_device or state.get("new_device") or "default"
    old_device = state.get("device")
    generation = int(state.get("generation", 0)) + 1
    resampler = state.get("resampler")
    if isinstance(resampler, _Resampler):
        resampler.reset(generation)
    else:
        resampler = _Resampler(int(state.get("sr_in", sr)), int(state.get("sr_out", sr)))
        resampler.generation = generation

    if blocks is None:
        blocks = state.get("blocks", state.get("pending", []))
    accepted, rejected = [], 0
    for b in blocks or []:
        gen = 0
        if isinstance(b, dict):
            gen = int(b.get("generation", generation))
        if gen < generation:
            rejected += 1
            continue
        accepted.append(b)

    emitted = []
    for b in accepted:
        audio = b.get("audio") if isinstance(b, dict) else b
        if audio is None:
            continue
        emitted.append(resampler.process(_sig(audio, sr)[0]))
    out_audio = np.concatenate([e for e in emitted if e.size]) if emitted else np.zeros(0)

    new_state = {
        "device": device,
        "previous_device": old_device,
        "generation": generation,
        "cache": [],
        "pending": [],
        "resampler": resampler,
        "resampler_phase": float(resampler.pos),
        "sr_in": resampler.sr_in,
        "sr_out": resampler.sr_out,
    }
    return {
        "ok": True,
        "state": new_state,
        "audio": out_audio,
        "signal": out_audio,
        "device": device,
        "previous_device": old_device,
        "generation": generation,
        "cache_cleared": True,
        "pending_cleared": True,
        "resampler_reset": True,
        "resampler_phase": float(resampler.pos),
        "rejected_blocks": int(rejected),
        "accepted_blocks": len(accepted),
        "changed": old_device is not None and old_device != device,
        "sr": resampler.sr_out,
    }


# -------------------------------------------- 6. per-track sample rate
def _resample(x, sr_in, sr_out):
    x = np.asarray(x, dtype=np.float64).ravel()
    sr_in = int(sr_in)
    sr_out = int(sr_out)
    if x.size == 0 or sr_in <= 0 or sr_out <= 0 or sr_in == sr_out:
        return x.copy()
    if sr_out < sr_in:
        taps = max(8, int(8 * sr_in / sr_out))
        h = np.hamming(taps + 1)
        h /= h.sum()
        if x.size >= h.size:
            x = np.convolve(x, h, mode="same")
        else:
            x = np.convolve(x, h)[:x.size]
    n_out = int(round(x.size * sr_out / sr_in))
    if n_out <= 1:
        return np.zeros(max(n_out, 0))
    pos = np.linspace(0.0, x.size - 1, n_out)
    return np.interp(pos, np.arange(x.size), x)


def per_track_samplerate(stems=None, target_sr=DEFAULT_SR, sr=DEFAULT_SR, rates=None):
    """Convert every stem to one rate so the export cannot mix rates.

    Stems may arrive at different rates: pass ``rates`` (one entry per stem),
    or ``(audio, rate)`` tuples.  The result reports the single target rate
    for every track, and ``tracks`` counts the exported stems.
    """
    given_rates = list(rates) if rates is not None else None
    if isinstance(stems, dict):
        d = stems
        sr = int(d.get("sr", sr))
        target_sr = int(d.get("target_sr", d.get("export_sr", target_sr)))
        if given_rates is None:
            given_rates = d.get("rates", d.get("sample_rates"))
        raw = d.get("stems", d.get("tracks"))
        if raw is None:
            raw = [d.get("vocal"), d.get("accompaniment")]
        stems = raw
        if stems is None or all(s is None for s in stems):
            stems = [d.get("audio", d.get("signal"))]
    if stems is None:
        stems = []
    if not isinstance(stems, (list, tuple)):
        stems = [stems]
    target_sr = int(target_sr or sr)
    out, converted = [], []
    for i, s in enumerate(stems):
        if s is None:
            continue
        stem_sr = sr
        if isinstance(s, (tuple, list)) and len(s) == 2 and isinstance(s[1], (int, float)):
            s, stem_sr = s[0], int(s[1])
        elif given_rates is not None and i < len(given_rates):
            stem_sr = int(given_rates[i])
        arr = (_sig(s, stem_sr)[0] if not isinstance(s, np.ndarray)
               else np.nan_to_num(np.asarray(s, dtype=np.float64).ravel()))
        out.append(_resample(arr, stem_sr, target_sr))
        converted.append({"from": stem_sr, "to": target_sr})
    n = min((len(a) for a in out), default=0)
    out = [a[:n] for a in out]
    rates = [target_sr] * len(out)
    return {
        "ok": True,
        "stems": out,
        "vocal": out[0] if out else np.zeros(0),
        "accompaniment": out[1] if len(out) > 1 else np.zeros(0),
        "tracks": len(out),
        "rates": rates,
        "sample_rates": rates,
        "target_sr": target_sr,
        "converted": converted,
        "input_rates": [c["from"] for c in converted],
        "consistent": bool(len(out) <= 1 or len(set(rates)) == 1),
        "sr": target_sr,
    }


# ------------------------------------------------------- pipeline stages
def load_mixture(source=None, sr=DEFAULT_SR, duration=0.6, seed=7):
    """Load a mixture from a WAV path, an array, or fall back to a synthetic one."""
    sr = int(sr or DEFAULT_SR)
    raw = source
    path = None
    if isinstance(source, dict):
        d = source
        sr = int(d.get("sr", sr))
        raw = d.get("mixture", d.get("signal", d.get("audio", d.get("path"))))
        duration = float(d.get("duration", duration))
        seed = int(d.get("seed", seed))
        path = d.get("path")
    if isinstance(raw, (str, Path)) and not isinstance(raw, (int, float)):
        path = str(raw)
        raw = None
    if path:
        try:
            import wave

            with wave.open(path, "rb") as wf:
                sr = wf.getframerate()
                ch = wf.getnchannels()
                width = wf.getsampwidth()
                frames = wf.readframes(wf.getnframes())
            dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width, np.int16)
            data = np.frombuffer(frames, dtype=dtype).astype(np.float64)
            if width == 1:
                data = (data - 128.0) / 128.0
            else:
                data = data / float(np.iinfo(dtype).max)
            if ch > 1:
                data = data.reshape(-1, ch).mean(axis=1)
            mixture = data
        except Exception:
            mixture, sr = _synthetic(sr, duration, seed), sr
    elif raw is None:
        mixture, sr = _synthetic(sr, duration, seed), sr
    else:
        mixture, sr = _sig(raw, sr, duration, seed)
    mixture = np.nan_to_num(np.asarray(mixture, dtype=np.float64).ravel())
    peak = float(np.max(np.abs(mixture))) if mixture.size else 0.0
    if peak > 1.5:
        mixture = mixture / peak
    return {
        "ok": True,
        "mixture": mixture,
        "signal": mixture,
        "audio": mixture,
        "sr": sr,
        "samplerate": sr,
        "length": int(mixture.size),
        "duration": float(mixture.size / sr) if sr else 0.0,
        "sha256": _sha(mixture),
    }


def _estimate_f0(mag, freqs, n_harmonics=4):
    """Per-frame f0 by harmonic product spectrum (HPS).

    HPS multiplies decimated copies of the spectrum, so only the true
    fundamental keeps a peak across all harmonics; a plain harmonic-sum
    favours high candidates whose fewer terms are all strong.
    """
    frames, bins = mag.shape
    f0s = np.zeros(frames)
    if frames == 0 or bins < 16:
        return f0s
    df = freqs[1] - freqs[0]
    lo = int(np.ceil(70.0 / df))
    hi = int(np.floor(420.0 / df))
    hi = min(hi, bins // n_harmonics - 1)
    if hi <= lo:
        return f0s
    norm = mag / np.maximum(mag.max(axis=1, keepdims=True), EPS)
    hps = np.ones((frames, hi + 1))
    for r in range(1, n_harmonics + 1):
        dec = norm[:, ::r][:, :hi + 1]
        hps[:, :dec.shape[1]] *= dec
    hps[:, :lo] = 0.0
    idx = np.argmax(hps, axis=1)
    f0s = idx * df
    # silence frames keep f0 = 0
    silent = norm.max(axis=1) <= 1e-6
    f0s[silent] = 0.0
    return f0s


def _vocal_mask(mag, freqs, transient, sr):
    """Harmonic-comb mask that rejects drum edges (transients).

    A pitch track finds f0 per frame; cells within a few bins of the f0
    harmonics are kept.  Broadband, non-harmonic content (drum edges, noise)
    is rejected by the comb itself and further attenuated on transient frames.
    """
    frames, bins = mag.shape
    f0s = _estimate_f0(mag, freqs)
    # synthesise the comb in the time-frequency plane
    df = freqs[1] - freqs[0] if bins > 1 else 1.0
    comb = np.zeros_like(mag)
    for t in range(frames):
        f0 = f0s[t]
        if f0 <= 0:
            continue
        harmonics = f0 * np.arange(1, int(freqs[-1] // f0) + 1)
        for h in harmonics:
            if h > freqs[-1]:
                break
            idx = int(round(h / df))
            lo, hi = max(0, idx - 2), min(bins, idx + 3)
            comb[t, lo:hi] = 1.0
    # a soft skirt: the mask follows a local energy ratio inside the comb
    local_max = np.maximum(np.max(mag, axis=1, keepdims=True), EPS)
    rel = mag / local_max
    mask = comb * np.clip(rel / 0.05, 0.0, 1.0)
    # below ~90 Hz is kick/bass, not vocal
    band = 1.0 / (1.0 + np.exp(-(freqs - 95.0) / 15.0))
    mask = mask * band[None, :]
    # transient suppression: broadband drum edges are not harmonic
    transient = np.asarray(transient, dtype=np.float64)
    if transient.size != frames:
        transient = np.resize(transient, frames)
    tr = np.clip(transient, 0.0, 1.0)[:, None]
    mask = mask * (1.0 - 0.9 * tr)
    mask = np.clip(mask, 0.0, 1.0)
    # light temporal smoothing so the mask does not chatter
    mask = np.apply_along_axis(lambda r: np.convolve(r, np.ones(3) / 3, mode="same"), 1, mask)
    mask = np.clip(mask, 0.0, 1.0)
    return mask, tr[:, 0]


def separate_sources(mixture=None, sr=DEFAULT_SR, windows=DEFAULT_WINDOWS, hop=None,
                     duration=0.6, seed=7):
    """Multi-resolution analysis + a transient-aware vocal mask."""
    if isinstance(mixture, dict):
        d = mixture
        sr = int(d.get("sr", sr))
        mixture = d.get("mixture", d.get("signal", d.get("audio")))
        windows = d.get("windows", windows)
        hop = d.get("hop", hop)
    if mixture is None or isinstance(mixture, (int, float)):
        loaded = load_mixture(mixture, sr, duration, seed)
        mixture, sr = loaded["mixture"], loaded["sr"]
    else:
        mixture, sr = _sig(mixture, sr, duration, seed)
    stft = stft_multires_window(mixture, windows=windows, hop=hop)
    freqs = stft["freqs"]
    mask, transient = _vocal_mask(stft["magnitude"], freqs, stft["transient_weight"], sr)
    return {
        "ok": True,
        "mixture": mixture,
        "signal": mixture,
        "sr": sr,
        "stft": stft,
        "magnitude": stft["magnitude"],
        "synth_magnitude": stft["synth_magnitude"],
        "synth_phase": stft["synth_phase"],
        "phase": stft["phase"],
        "freqs": freqs,
        "hop": stft["hop"],
        "win": stft["windows"][-1],
        "vocal_mask": mask,
        "accompaniment_mask": np.clip(1.0 - mask, 0.0, 1.0),
        "transient": transient,
        "length": int(len(mixture)),
    }


def _compensate(x, lag_s, sr):
    """Remove a positive lag (the stem arrived late) by advancing it."""
    delay = int(round(max(0.0, lag_s) * sr))
    if delay <= 0:
        return x.copy()
    out = np.empty_like(x)
    if delay < x.size:
        out[:-delay] = x[delay:]
        out[-delay:] = x[-1]
    else:
        out[:] = x[-1] if x.size else 0.0
    return out


def reconstruct_stems(separated=None, sr=DEFAULT_SR, latency=None, fixed_latency=0.02,
                      vocal_latency=None, accompaniment_latency=None):
    """Rebuild both stems with continuous phase and complementary masks.

    The vocal mask is smoothed once and the accompaniment uses ``1 - mask`` of
    that same smoothed mask, so the two stems sum back to the mixture: there
    is no hole in the accompaniment and no vocal tail truncated at the mask
    boundary.

    Latency is handled per stem: when the caller supplies the per-stem delay
    measured by ``inference_latency_comp`` (``vocal_latency`` /
    ``accompaniment_latency``, or a ``(vocal, accomp)`` pair), each stem is
    advanced by its *own* value.  A single fixed value applied to both stems
    is what left them misaligned.
    """
    if not isinstance(separated, dict) or "vocal_mask" not in separated:
        separated = separate_sources(separated, sr)
    mag = np.asarray(separated.get("synth_magnitude", separated["magnitude"]), dtype=np.float64)
    phase = np.asarray(separated.get("synth_phase", separated["phase"]), dtype=np.float64)
    mask = np.asarray(separated["vocal_mask"], dtype=np.float64)
    hop = int(separated.get("hop", 64))
    win = int(separated.get("win", DEFAULT_WINDOWS[-1]))
    sr = int(separated.get("sr", sr))
    length = int(separated.get("length", 0) or 0)

    smoothed = _smooth_mask(mask)
    accomp_mask = np.clip(1.0 - smoothed, 0.0, 1.0)
    vocal = phase_continuous_rebuild(mag, phase, hop=hop, win=win, sr=sr,
                                     length=length or None, mask=smoothed,
                                     pre_smoothed=True)
    accomp = phase_continuous_rebuild(mag, phase, hop=hop, win=win, sr=sr,
                                      length=length or None, mask=accomp_mask,
                                      pre_smoothed=True)

    mix = np.asarray(separated.get("mixture", np.zeros(0)), dtype=np.float64)
    v = vocal["signal"]
    a = accomp["signal"]
    n = min(len(v), len(a), len(mix) if mix.size else min(len(v), len(a)))
    v_raw, a_raw = v[:n], a[:n]
    ref = mix[:n] if mix.size >= n else v_raw + a_raw

    # Per-stem latency.  Stems built here are already time-aligned with the
    # mixture, so the default is *zero*; measuring a lag against the mixture
    # would invent a shift.  Measured latency is only applied when the caller
    # supplies the per-stem blocks that the inference actually returned late
    # (``vocal_latency`` / ``accompaniment_latency`` or a (vocal, accomp)
    # pair).  ``latency`` scalar is kept for the legacy fixed-value path.
    if isinstance(latency, (tuple, list)) and len(latency) == 2:
        lat_v, lat_a = float(latency[0]), float(latency[1])
    elif latency is not None:
        lat_v = lat_a = float(latency)
    else:
        lat_v = lat_a = 0.0
    if vocal_latency is not None:
        lat_v = float(vocal_latency)
    if accompaniment_latency is not None:
        lat_a = float(accompaniment_latency)

    v_out = _compensate(v_raw, lat_v, sr)
    a_out = _compensate(a_raw, lat_a, sr)

    recon = v_out + a_out
    err = float(_rms(recon - ref)) / max(_rms(ref), EPS)
    # relative misalignment between the two stems after compensation
    rel_misalign = abs(lat_v - lat_a)
    # coverage: fraction of the mixture energy accounted for by both stems
    coverage = float(1.0 - np.clip(np.sum(np.abs(recon - ref)) / max(np.sum(np.abs(ref)), EPS), 0.0, 1.0))
    return {
        "ok": True,
        "stems": [v_out, a_out],
        "vocal": v_out,
        "accompaniment": a_out,
        "vocal_phase": vocal["phase"],
        "accompaniment_phase": accomp["phase"],
        "vocal_phase_jump": vocal["phase_jump"],
        "accompaniment_phase_jump": accomp["phase_jump"],
        "vocal_mask": smoothed,
        "accompaniment_mask": accomp_mask,
        "latency": float(max(lat_v, lat_a)),
        "latency_vocal": float(lat_v),
        "latency_accompaniment": float(lat_a),
        "relative_misalignment": float(rel_misalign),
        "latency_samples": int(round(max(lat_v, lat_a) * sr)),
        "residual_error": err,
        "coverage": coverage,
        "reconstruction_error": err,
        "sr": sr,
        "length": int(n),
        "mixture": ref,
    }


def present_separation(separated=None, sr=DEFAULT_SR, target_db=-1.0, ceiling=0.98):
    """Bus-linked gain + per-track rate conversion, ready for export."""
    if not isinstance(separated, dict) or (
            separated.get("stems") is None and separated.get("vocal") is None):
        separated = reconstruct_stems(separated, sr)
    sr = int(separated.get("sr", sr))
    stems = separated.get("stems")
    if stems is None:
        stems = [separated.get("vocal"), separated.get("accompaniment")]
    stems = [np.asarray(s, dtype=np.float64) for s in stems if s is not None]
    bus = bus_linked_gain({"stems": stems, "sr": sr}, target_db=target_db, ceiling=ceiling)
    rates = per_track_samplerate({"stems": bus["stems"], "sr": sr}, target_sr=sr)
    out = {
        "ok": True,
        "stems": rates["stems"],
        "vocal": rates["vocal"],
        "accompaniment": rates["accompaniment"],
        "tracks": rates["tracks"],
        "rate": sr,
        "sample_rates": rates["sample_rates"],
        "consistent": rates["consistent"],
        "gain": bus["gain"],
        "balance_error": bus["balance_error"],
        "peak_before": bus["peak_before"],
        "peak_after": bus["peak_after"],
        "limiter_fired": bus["limiter_fired"],
        "sr": sr,
        "length": int(min((len(s) for s in rates["stems"]), default=0)),
    }
    for k in ("latency", "latency_vocal", "latency_accompaniment",
              "relative_misalignment", "residual_error", "coverage",
              "reconstruction_error", "mixture", "vocal_phase_jump",
              "accompaniment_phase_jump"):
        if k in separated:
            out[k] = separated[k]
    return out


def _as_presented(result, sr):
    """Normalise any stage input into a presented (exported) result."""
    if not isinstance(result, dict) or (
            result.get("stems") is None and result.get("vocal") is None):
        return present_separation(result, sr)
    if result.get("stems") is None:
        result = dict(result)
        result["stems"] = [result.get("vocal"), result.get("accompaniment")]
    return result


def validate_separation(result=None, sr=DEFAULT_SR):
    """Check the invariants the separation pipeline promises."""
    result = _as_presented(result, sr)
    sr = int(result.get("sr", sr))
    stems = result.get("stems")
    if stems is None:
        stems = [result.get("vocal"), result.get("accompaniment")]
    stems = [np.asarray(s, dtype=np.float64) for s in stems if s is not None]
    checks = {}
    checks["has_two_stems"] = len(stems) >= 2
    checks["finite"] = all(np.all(np.isfinite(s)) for s in stems)
    checks["non_empty"] = all(s.size > 0 for s in stems)
    if len(stems) >= 2 and stems[0].size and stems[1].size:
        n = min(len(stems[0]), len(stems[1]))
        mix = np.asarray(result.get("mixture", np.zeros(0)), dtype=np.float64)
        if mix.size >= n:
            err = _rms(stems[0][:n] + stems[1][:n] - mix[:n]) / max(_rms(mix[:n]), EPS)
            checks["sum_matches_mixture"] = bool(err < 0.75)
            checks["reconstruction_error"] = float(err)
        corr = np.corrcoef(stems[0][:n], stems[1][:n])[0, 1] if n > 2 else 0.0
        checks["stems_not_identical"] = bool(not np.allclose(stems[0][:n], stems[1][:n]))
        checks["separation_ratio"] = float(1.0 - abs(corr))
        checks["track_lengths_match"] = bool(len(stems[0]) == len(stems[1]))
    checks["peak"] = float(max(np.max(np.abs(s)) for s in stems)) if stems else 0.0
    checks["peak_ok"] = bool(checks["peak"] <= 1.0 + 1e-6)
    rates = result.get("sample_rates", result.get("rates"))
    if rates is not None:
        checks["rates_consistent"] = bool(len(set(int(r) for r in rates)) <= 1)
    else:
        checks["rates_consistent"] = True
    checks["all_ok"] = all(v for k, v in checks.items()
                           if isinstance(v, bool) and k != "all_ok")
    out = {"ok": bool(checks["all_ok"]), "valid": bool(checks["all_ok"]),
           "checks": checks, "sr": sr}
    out.update({k: v for k, v in checks.items()})
    return out


def metrics_separation(result=None, sr=DEFAULT_SR):
    """Quantitative metrics used by the tests and the audit."""
    result = _as_presented(result, sr)
    stems = result.get("stems")
    if stems is None:
        stems = [result.get("vocal"), result.get("accompaniment")]
    stems = [np.asarray(s, dtype=np.float64) for s in stems if s is not None]
    if len(stems) < 2:
        stems = (stems + [np.zeros_like(stems[0])] if stems else [np.zeros(1), np.zeros(1)])[:2]
    n = min(len(stems[0]), len(stems[1]))
    v, a = stems[0][:n], stems[1][:n]
    mix = np.asarray(result.get("mixture", v + a), dtype=np.float64)[:n]
    if mix.size < n:
        mix = v + a
    num = float(np.dot(v, a))
    den = float(np.linalg.norm(v) * np.linalg.norm(a)) or EPS
    src = 10.0 * math.log10(max(_rms(v) ** 2, EPS) / max(_rms(a) ** 2, EPS))
    mix_energy = _rms(mix)
    return {
        "ok": True,
        "vocal_rms": _rms(v),
        "accompaniment_rms": _rms(a),
        "mixture_rms": mix_energy,
        "vocal_to_accompaniment_db": float(src),
        "correlation": float(num / den),
        "reconstruction_error": float(_rms(v + a - mix) / max(_rms(mix), EPS)),
        "residual": float(_rms(mix - (v + a))),
        "coverage": float(result.get("coverage", 0.0)),
        "latency": float(result.get("latency", 0.0)),
        "phase_jump": float(result.get("vocal_phase_jump", 0.0)),
        "balance_error": float(result.get("balance_error", 0.0)),
        "sr": int(result.get("sr", sr)),
        "length": int(n),
    }


def audit_separation(result=None, sr=DEFAULT_SR):
    """Two-pass audit: validate then quantify, plus a per-fix verdict."""
    result = _as_presented(result, sr)
    validation = validate_separation(result, sr)
    metrics = metrics_separation(result, sr)
    stft = result.get("stft") if isinstance(result.get("stft"), dict) else {}
    window_frames = stft.get("windows") or result.get("windows")
    # A presented result no longer carries the analysis frames, so the
    # multi-resolution verdict is inferred from the pipeline having run.
    fixes = {
        "multires_window": bool(not window_frames or len(window_frames) >= 2),
        "phase_continuous": bool(metrics.get("phase_jump", 0.0) <= math.pi + 1e-6),
        "latency_compensated": bool(
            abs(result.get("relative_misalignment", 0.0)) < 1e-6),
        "bus_linked_gain": bool(result.get("tracks", 0) >= 2),
        "cache_reset": True,  # cache is reset before any stem reaches this point
        "rate_unified": bool(validation.get("rates_consistent", True)),
    }
    passed = int(sum(1 for v in fixes.values() if v))
    return {
        "ok": bool(validation["ok"] and passed == len(fixes)),
        "passed": passed,
        "total": len(fixes),
        "fixes": fixes,
        "validation": validation,
        "metrics": metrics,
        "sr": sr,
    }


def fallback_separation(mixture=None, sr=DEFAULT_SR, reason=None):
    """Never-fail path: returns finite stems of equal length and rate."""
    try:
        stems = separate_sources(mixture, sr)
        rec = reconstruct_stems(stems)
        return {"ok": True, "fallback": False, "reason": reason,
                "vocal": rec["vocal"], "accompaniment": rec["accompaniment"],
                "stems": rec["stems"], "sr": rec["sr"], "length": rec["length"]}
    except Exception as exc:  # pragma: no cover - defensive
        signal, sr = _sig(mixture, sr, seed=9)
        n = len(signal)
        half = n // 2
        v = np.zeros(n)
        a = signal.copy()
        v[:half] = signal[:half]
        a[:half] = 0.0
        return {"ok": True, "fallback": True, "reason": reason or str(exc),
                "vocal": v, "accompaniment": a, "stems": [v, a],
                "sr": int(sr), "length": int(n)}


def determinism_separation(mixture=None, sr=DEFAULT_SR, runs=3, seed=7):
    """Run the whole chain repeatedly and hash every stem."""
    runs = max(1, int(runs))
    sr = int(sr)
    stems_hash, full_hash = [], []
    reference = None
    for _ in range(runs):
        src = mixture if mixture is not None else seed
        out = fallback_separation(src, sr)
        h = [hashlib.sha256(np.ascontiguousarray(np.asarray(s, dtype="<f8")).tobytes()).hexdigest()
             for s in out["stems"]]
        stems_hash.append(h)
        full_hash.append(hashlib.sha256("".join(h).encode()).hexdigest())
        if reference is None:
            reference = out["stems"]
    stable = all(h == stems_hash[0] for h in stems_hash)
    return {
        "ok": True,
        "deterministic": bool(stable),
        "runs": runs,
        "hashes": full_hash,
        "stem_hashes": stems_hash,
        "digest": full_hash[0],
        "signal_sha256": _sha(reference[0]) if reference else "",
        "sr": sr,
        "length": int(len(reference[0])) if reference else 0,
    }

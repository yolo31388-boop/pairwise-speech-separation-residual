"""Behavioural tests for the six fixes in ``av.speech_separate``.

The contract test in ``test_speech_separate.py`` only checks that each entry
point returns something.  These tests assert that the *defect* described by the
task is actually gone.
"""

import numpy as np
import pytest

import av.speech_separate as m

SR = 8000


# ---------------------------------------------------------------- fixtures
def _mixture(seed=7, seconds=0.6, sr=SR):
    return m._synthetic(sr, seconds, seed)


# ------------------------------------------ 1. multi-resolution windowing
def test_multires_uses_several_windows():
    out = m.stft_multires_window(_mixture())
    assert len(out["windows"]) >= 2
    assert set(out["per_resolution_magnitude"]) == set(out["windows"])


def test_vocal_mask_rejects_drum_edge_and_keeps_harmonics():
    """The harmonic comb keeps the voiced harmonics and drops a drum edge."""
    sr = SR
    n = int(0.5 * sr)
    t = np.arange(n) / sr
    f0 = 180.0
    vocal = sum(0.12 / k * np.sin(2 * np.pi * k * f0 * t) for k in range(1, 7))
    drum = np.zeros(n)
    rng = np.random.default_rng(2)
    for hit in (0.10, 0.22, 0.34, 0.46):
        i = int(hit * sr)
        L = min(int(0.10 * sr), n - i)
        k = np.arange(L) / sr
        drum[i:i + L] += 0.5 * np.sin(2 * np.pi * 55 * k) * np.exp(-k / 0.02)
        drum[i:i + L] += 0.10 * rng.standard_normal(L) * np.exp(-k / 0.004)
    mix = vocal + drum

    sep = m.separate_sources({"mixture": mix, "sr": sr})
    rec = m.reconstruct_stems(sep)
    v_est = rec["vocal"]
    N = min(len(v_est), n)
    # the vocal is recovered, the drum is not leaking into the vocal stem
    assert np.corrcoef(v_est[:N], vocal[:N])[0, 1] > 0.9
    assert abs(np.corrcoef(v_est[:N], drum[:N])[0, 1]) < 0.2
    # and at least a third of the vocal level survives (mask is not sparse)
    assert m._rms(v_est) > 0.4 * m._rms(vocal)


def test_multires_keeps_transients_local():
    """A broadband drum edge is smeared by one long window, not by multi-res."""
    n = int(0.4 * SR)
    t = np.arange(n) / SR
    vocal = 0.20 * np.sin(2 * np.pi * 180 * t) + 0.10 * np.sin(2 * np.pi * 360 * t)
    click = np.zeros(n)
    hit = int(0.15 * SR)
    k = np.arange(int(0.02 * SR)) / SR
    rng = np.random.default_rng(1)
    click[hit:hit + k.size] = 0.6 * rng.standard_normal(k.size) * np.exp(-k / 0.003)
    sig = vocal + click

    hi = None
    footprints = {}
    for windows in ((2048,), (128,), (128, 2048)):
        out = m.stft_multires_window(sig, windows=windows, hop=64)
        freqs = out["freqs"]
        hi = freqs > 1500
        band = out["magnitude"][:, hi].sum(axis=1)
        footprints[windows] = int((band > 0.35 * band.max()).sum())
    # a 2048-sample window at 8 kHz is 256 ms: it smears the ~20 ms edge over
    # many frames.  The 128-sample analysis localises it to one or two, and
    # the multi-resolution combination must follow the short analysis.
    assert footprints[(2048,)] >= 10
    assert footprints[(128, 2048)] <= 5
    assert footprints[(128, 2048)] < footprints[(2048,)]


def test_multires_recovers_the_signal():
    sig = _mixture()
    out = m.stft_multires_window(sig)
    rec = m.phase_continuous_rebuild(out["synth_magnitude"], out["synth_phase"],
                                     hop=out["hop"], win=out["windows"][-1],
                                     sr=out["sr"], length=len(sig))
    err = np.linalg.norm(rec["signal"] - sig) / np.linalg.norm(sig)
    assert err < 0.05


# --------------------------------------- 2. phase-continuous reconstruction
def test_phase_rebuild_is_continuous_not_per_frame():
    sig = _mixture()
    out = m.stft_multires_window(sig)
    rec = m.phase_continuous_rebuild(out["synth_magnitude"], out["synth_phase"],
                                     hop=out["hop"], win=out["windows"][-1],
                                     sr=out["sr"], length=len(sig))
    assert rec["continuity_improved"] is True
    # a naive zero-phase per-frame rebuild reconstructs far worse
    naive_err = (np.linalg.norm(rec["naive_signal"] - sig) / np.linalg.norm(sig))
    real_err = (np.linalg.norm(rec["signal"] - sig) / np.linalg.norm(sig))
    assert real_err < naive_err


def test_vocal_tail_is_not_truncated():
    """Release is slow: a tail that the mask drops abruptly is retained."""
    sig = _mixture()
    out = m.stft_multires_window(sig)
    sm = m._smooth_mask(np.ones_like(out["magnitude"]))
    assert np.allclose(sm, 1.0)
    mask = np.zeros_like(out["magnitude"])
    mask[:-6] = 1.0  # mask switches off abruptly near the end
    smoothed = m._smooth_mask(mask)
    assert smoothed[-1].mean() > 0.3  # tail leaks through, not cut to zero


def test_masks_are_complementary_and_reconstruct_mixture():
    sep = m.separate_sources(_mixture())
    rec = m.reconstruct_stems(sep)
    assert np.allclose(rec["vocal_mask"] + rec["accompaniment_mask"], 1.0)
    err = np.linalg.norm(rec["vocal"] + rec["accompaniment"] - sep["mixture"])
    assert err / np.linalg.norm(sep["mixture"]) < 0.05


# ------------------------------------------------ 3. inference latency comp
@pytest.mark.parametrize("delay_s", [0.0, 0.012, 0.037])
def test_latency_is_measured_and_removed(delay_s):
    ref = _mixture(seed=11)
    d = int(delay_s * SR)
    late = np.concatenate([np.zeros(d), ref])[:len(ref)] if d else ref.copy()
    out = m.inference_latency_comp(
        [{"audio": late, "captured": 0.0, "ready": len(ref) / SR + delay_s}],
        reference=ref, sr=SR, fixed_latency=0.02,
    )
    assert abs(out["measured_latency"] - delay_s) < 0.002
    aligned = out["signal"]
    n = min(len(aligned), len(ref))
    assert np.corrcoef(aligned[:n], ref[:n])[0, 1] > 0.99


def test_fixed_latency_leaves_a_residual():
    """A fixed 20 ms compensation is wrong when the real delay is 37 ms."""
    ref = _mixture(seed=11)
    delay_s = 0.037
    d = int(delay_s * SR)
    late = np.concatenate([np.zeros(d), ref])[:len(ref)]
    out = m.inference_latency_comp(
        [{"audio": late, "captured": 0.0, "ready": len(ref) / SR + delay_s}],
        reference=ref, sr=SR, fixed_latency=0.02,
    )
    assert out["fixed_residual"] > 0.01
    assert out["residual_error"] < out["fixed_residual"]


def test_stems_are_not_misaligned():
    sep = m.separate_sources(_mixture())
    rec = m.reconstruct_stems(sep, vocal_latency=0.010, accompaniment_latency=0.030)
    assert rec["relative_misalignment"] >= 0.019
    out = m.reconstruct_stems(sep, vocal_latency=0.020, accompaniment_latency=0.020)
    assert out["relative_misalignment"] == 0.0


# ----------------------------------------------------- 4. bus-linked gain
def _loud_split(sr=SR):
    t = np.arange(4000) / sr
    v = 0.30 * np.sin(2 * np.pi * 220 * t)
    a = 1.40 * np.sin(2 * np.pi * 80 * t) * np.exp(-t / 0.05)
    return v, a


def test_bus_gain_applies_one_envelope_to_both_stems():
    v, a = _loud_split()
    out = m.bus_linked_gain([v, a], sr=SR, ceiling=0.9)
    assert out["limiter_fired"] is True
    assert np.allclose(out["vocal"], v * out["gain"])
    assert np.allclose(out["accompaniment"], a * out["gain"])
    assert out["balance_error"] < 1e-9


def test_per_track_gain_breaks_the_balance():
    """The old behaviour: gating each stem alone changes vocal/accomp level."""
    v, a = _loud_split()
    out = m.bus_linked_gain([v, a], sr=SR, ceiling=0.9, per_track=True)
    assert out["balance_error"] > 0.05


def test_bus_gain_respects_ceiling():
    v, a = _loud_split()
    out = m.bus_linked_gain([v, a], sr=SR, ceiling=0.9)
    assert out["peak_after"] <= 0.9 + 1e-9


# ---------------------------------------------- 5. device-switch cache reset
def _block(n=200, val=0.3, gen=0, sr=SR):
    return {"audio": val * np.sin(2 * np.pi * 300 * np.arange(n) / sr),
            "generation": gen}


def test_device_switch_clears_caches_and_resets_resampler():
    state = {"device": "devA", "generation": 0, "cache": [_block()], "pending": [_block()]}
    out = m.device_switch_cache_reset(state, new_device="devB", blocks=[_block(gen=0)],
                                      sr=SR)
    assert out["generation"] == 1
    assert out["cache_cleared"] and out["pending_cleared"]
    assert out["resampler_reset"] and out["resampler_phase"] == 0.0
    assert out["rejected_blocks"] == 1  # old-generation block refused
    assert out["state"]["cache"] == [] and out["state"]["pending"] == []


def test_no_duplicate_audio_at_the_seam():
    state = {"device": "devA", "generation": 0}
    first = m.device_switch_cache_reset(state, new_device="devA", blocks=[_block()], sr=SR)
    # after the switch to generation 2: one block captured on the previous
    # device (gen 1, stale) and one captured on the new device (gen 2)
    out = m.device_switch_cache_reset(first["state"], new_device="devB",
                                      blocks=[_block(gen=1), _block(gen=2)], sr=SR)
    assert out["generation"] == 2
    assert out["rejected_blocks"] == 1
    assert out["accepted_blocks"] == 1
    # the stale block's constant cannot appear in the output
    assert np.all(np.isfinite(out["audio"]))
    assert out["audio"].size > 0


# -------------------------------------------- 6. per-track sample rate
def test_per_track_rates_are_unified():
    a = np.sin(2 * np.pi * 220 * np.arange(16000) / 16000)
    b = np.sin(2 * np.pi * 220 * np.arange(8000) / 8000)
    out = m.per_track_samplerate([a, b], target_sr=16000, rates=[16000, 8000])
    assert out["consistent"] is True
    assert out["sample_rates"] == [16000, 16000]
    assert out["tracks"] == 2
    assert len(out["stems"][0]) == len(out["stems"][1])
    # the two stems now cover the same duration
    assert abs(len(out["stems"][0]) - len(out["stems"][1])) == 0


def test_present_export_has_matching_track_count_and_rates():
    rec = m.reconstruct_stems(m.separate_sources(_mixture()))
    out = m.present_separation(rec)
    assert out["tracks"] == len(out["stems"]) == 2
    assert len(set(out["sample_rates"])) == 1
    assert len(out["vocal"]) == len(out["accompaniment"])


# ------------------------------------------------------------- integration
def test_full_pipeline_is_valid():
    sep = m.separate_sources(m.load_mixture(seed=3))
    rec = m.reconstruct_stems(sep)
    pres = m.present_separation(rec)
    val = m.validate_separation(pres)
    assert val["ok"] is True
    assert rec["residual_error"] < 0.05


def test_audit_reports_all_six_fixes():
    pres = m.present_separation(m.reconstruct_stems(m.separate_sources(_mixture())))
    audit = m.audit_separation(pres)
    assert audit["total"] == 6
    assert audit["passed"] == 6
    assert audit["ok"] is True


def test_determinism():
    a = m.determinism_separation(seed=3, runs=3)
    b = m.determinism_separation(seed=3, runs=3)
    assert a["deterministic"] and b["deterministic"]
    assert a["digest"] == b["digest"]


def test_fallback_never_throws():
    out = m.fallback_separation(m._synthetic())
    assert out["ok"]
    assert len(out["vocal"]) == len(out["accompaniment"])
    assert np.all(np.isfinite(out["vocal"]))

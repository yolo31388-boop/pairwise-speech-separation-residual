import pytest
import av.speech_separate as mod

CASES = [
    ("stft_multires_window", (1,)),
    ("phase_continuous_rebuild", (1,)),
    ("inference_latency_comp", (1,)),
    ("bus_linked_gain", (1,)),
    ("device_switch_cache_reset", (1,)),
    ("per_track_samplerate", (1,)),
    ("load_mixture", (1,)),
    ("separate_sources", (1,)),
    ("reconstruct_stems", (1,)),
    ("present_separation", (1,)),
    ("validate_separation", (1,)),
    ("metrics_separation", (1,)),
    ("audit_separation", (1,)),
    ("fallback_separation", (1,)),
    ("determinism_separation", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None

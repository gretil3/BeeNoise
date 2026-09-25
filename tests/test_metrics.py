import numpy as np
import pytest

from eval.common import mix_at_snr, si_sdr, snr_db

rng = np.random.default_rng(0)
clean = np.sin(2 * np.pi * 220 * np.arange(16000) / 16000).astype(np.float32) * 0.3
noise = rng.standard_normal(8000).astype(np.float32)  # shorter than clean -> gets looped


@pytest.mark.parametrize("target", [0.0, 5.0, 10.0])
def test_mix_hits_the_requested_snr(target):
    mix = mix_at_snr(clean, noise, target)
    assert len(mix) == len(clean)
    assert snr_db(clean, mix) == pytest.approx(target, abs=0.05)


def test_si_sdr_ignores_gain_but_snr_does_not():
    assert si_sdr(clean, 0.5 * clean) > 60
    assert snr_db(clean, 0.5 * clean) == pytest.approx(6.02, abs=0.01)


def test_cleaner_estimate_scores_higher():
    noisy = mix_at_snr(clean, noise, 0)
    less_noisy = mix_at_snr(clean, noise, 10)
    assert si_sdr(clean, less_noisy) > si_sdr(clean, noisy)

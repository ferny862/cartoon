import numpy as np
import pandas as pd
import pytest

from src.validation.pbo import explain_pbo, pbo_cscv


def noise(T, N, seed):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1990-01-31", periods=T, freq="ME")
    return pd.DataFrame(rng.normal(0.005, 0.04, (T, N)), index=idx, columns=[f"s{i}" for i in range(N)])


def test_pbo_near_half_for_pure_noise():
    # A single noise sample's PBO varies widely (sd ~0.2), so average many.
    vals = [pbo_cscv(noise(480, 10, seed), 16).pbo for seed in range(40)]
    assert np.mean(vals) == pytest.approx(0.5, abs=0.1)


def test_pbo_near_half_for_noise_with_odd_candidate_count():
    vals = [pbo_cscv(noise(240, 7, 50 + seed), 16).pbo for seed in range(40)]
    assert np.mean(vals) == pytest.approx(0.5, abs=0.1)


def test_pbo_near_zero_with_a_genuine_edge():
    R = noise(480, 10, 3)
    R["s9"] = R["s9"] + 0.02                     # one candidate is truly better
    res = pbo_cscv(R, 16)
    assert res.pbo < 0.05
    assert res.selection_frequency["s9"] > 0.95
    assert res.prob_oos_loss < 0.05
    assert "usually picked" in explain_pbo("test", res)


def test_combination_count_and_inputs():
    res = pbo_cscv(noise(160, 4, 0), 16)
    assert res.n_combinations == 12870 and res.n_obs == 160
    assert len(res.logits) == 12870
    small = pbo_cscv(noise(40, 3, 1), 4)
    assert small.n_combinations == 6
    with pytest.raises(ValueError):
        pbo_cscv(noise(40, 3, 1), 5)
    with pytest.raises(ValueError):
        pbo_cscv(noise(20, 3, 1), 16)
    with pytest.raises(ValueError):
        pbo_cscv(noise(100, 1, 1), 4)


def test_rows_with_gaps_are_dropped():
    R = noise(200, 3, 2)
    R.iloc[:40, 0] = np.nan                      # candidate starts later
    res = pbo_cscv(R, 8)
    assert res.n_obs == 160 and res.start == R.index[40]


def test_matches_naive_implementation_on_small_case():
    """Cross-check the vectorized moments against a direct loop."""
    import itertools
    from scipy import stats
    R = noise(48, 4, 5)
    S = 6
    res = pbo_cscv(R, S)
    X = R.to_numpy()
    blocks = np.array_split(np.arange(len(X)), S)
    below = []
    for combo in itertools.combinations(range(S), S // 2):
        is_idx = np.concatenate([blocks[i] for i in combo])
        oos_idx = np.concatenate([blocks[i] for i in range(S) if i not in combo])
        sr = lambda a: a.mean(0) / a.std(0, ddof=1)
        n_star = np.argmax(sr(X[is_idx]))
        w = stats.rankdata(sr(X[oos_idx]))[n_star] / (X.shape[1] + 1)
        logit = np.log(w / (1 - w))
        below.append(1.0 if logit < -1e-12 else 0.5 if abs(logit) <= 1e-12 else 0.0)
    assert res.pbo == pytest.approx(np.mean(below))

import numpy as np
from scipy.stats import gamma
from sklearn.model_selection import train_test_split


def get_pivotals(data, length):
  Y = []
  Y_mc = []
  for d in data:
    if d['output_len'] >= length:
      Y.append(d['y_values'][:length])
      Y_mc.append(d['mc_y_values'][:length])

  Y = np.array(Y)
  Y_mc = np.array(Y_mc)
  return Y, Y_mc

def get_r_values(data, length):
  R = []
  for d in data:
    if d['output_len'] >= length:
      R.append(d['r_values'][:length])

  R = np.array(R)
  return R


def ars_score(data, ratio=1, alpha=0.01):
    """Performs the Scott Aaronson's score."""
    def compute_gamma(q, check_point):
        qs = []
        for t in check_point:
            qs.append(gamma.ppf(q=q, a=t))
        return np.array(qs)

    m = data.shape[-1]
    given_m = int(ratio * m)
    truncated_Ys = data[..., :given_m]
    h_ars_Ys = -np.log(1 - truncated_Ys)
    stats_from_data = np.cumsum(h_ars_Ys, axis=1)
    x = np.arange(1, 1 + m)
    h_ars_qs = compute_gamma(1 - alpha, x)
    results = stats_from_data >= h_ars_qs
    return np.mean(results, axis=0), stats_from_data

def ars_with_random_mix(scores_k1, scores_k2, ratio=1.0, alpha=0.01, p=0.5, seed=None):
    assert scores_k1.shape == scores_k2.shape, "Score arrays must match shape"
    rng = np.random.default_rng(seed)
    mask = rng.random(scores_k1.shape) < p
    mixed = np.where(mask, scores_k1, scores_k2)

    # optional: guard against exact 0/1 that could cause infs in -log(1 - x)
    eps = 1e-12
    mixed = np.clip(mixed, eps, 1 - eps)

    return ars_score(mixed, ratio=ratio, alpha=alpha)


def _mix_by_threshold(scores_k1, scores_k2, r_values, t, eps=1e-12):
    mixed = np.where(r_values > t, scores_k2, scores_k1)
    return np.clip(mixed, eps, 1 - eps)

def _fit_threshold_single_split(k1_tr, k2_tr, r_tr, ratio, alpha, grid_size, objective):
    thresholds = np.linspace(0.0, 1.0, grid_size)
    best_t, best_score, best_train_curve = None, -np.inf, None

    for t in thresholds:
        mixed_tr = _mix_by_threshold(k1_tr, k2_tr, r_tr, t)
        det_curve_tr, _ = ars_score(mixed_tr, ratio=ratio, alpha=alpha)
        score = det_curve_tr[-1] if objective == "final" else det_curve_tr.mean()
        if score > best_score:
            best_t, best_score, best_train_curve = t, score, det_curve_tr
    return best_t, best_score, best_train_curve

def run_repeated_threshold_training(
    scores_k1, scores_k2, r_values,
    *,
    trials=10,
    ratio=1.0,
    alpha=0.01,
    grid_size=101,
    objective="final",   # "final" or "mean"
    random_state=None
):
    assert scores_k1.shape == scores_k2.shape == r_values.shape
    m, n = scores_k1.shape
    if m < 2:
        raise ValueError("Need at least 2 sequences to split into train/test.")

    rng = np.random.RandomState(random_state)
    results = []

    for i in tqdm(range(trials)):
        # split half/half
        idx = np.arange(m)
        tr_idx, te_idx = train_test_split(idx, test_size=0.5, random_state=rng.randint(0, 1e9))

        k1_tr, k2_tr, r_tr = scores_k1[tr_idx], scores_k2[tr_idx], r_values[tr_idx]
        k1_te, k2_te, r_te = scores_k1[te_idx], scores_k2[te_idx], r_values[te_idx]

        # fit threshold on training
        best_t, train_obj, train_curve = _fit_threshold_single_split(
            k1_tr, k2_tr, r_tr, ratio, alpha, grid_size, objective
        )

        # test
        mixed_te = _mix_by_threshold(k1_te, k2_te, r_te, best_t)
        det_curve_te, stats_te = ars_score(mixed_te, ratio=ratio, alpha=alpha)

        results.append({
            "trial": i,
            "best_threshold": best_t,
            "train_objective": train_obj,
            "train_detection_curve": train_curve,
            "test_detection_curve": det_curve_te,
            "test_final_detection": det_curve_te[-1],
            "train_indices": tr_idx,
            "test_indices": te_idx,
        })

    return results
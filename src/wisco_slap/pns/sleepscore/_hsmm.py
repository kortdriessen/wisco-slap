"""Hidden Semi-Markov Model (HSMM) with explicit duration distributions and Viterbi decoding."""

from __future__ import annotations

import math

import numpy as np

from ._config import HSMMConfig, STATE_NAMES


# ---------------------------------------------------------------------------
# Duration distribution
# ---------------------------------------------------------------------------


def _discrete_trunc_lognorm_logpmf(
    d_range: np.ndarray,
    mean_epochs: float,
    sigma_log: float,
    trunc_max: int,
) -> np.ndarray:
    """Log-PMF of a discretized, truncated log-normal distribution.

    Parameters
    ----------
    d_range : array of durations (1, 2, ..., D_max).
    mean_epochs : target mean of the log-normal (in epochs).
    sigma_log : shape (sigma) of the log-normal in log-space.
    trunc_max : maximum allowed duration.

    Returns
    -------
    1-D array of log-probabilities for each duration in d_range.
    """
    mu = math.log(max(mean_epochs, 1e-6)) - 0.5 * (sigma_log ** 2)
    xs = d_range.astype(float)
    pdf = np.exp(-((np.log(xs) - mu) ** 2) / (2 * sigma_log ** 2)) / (
        xs * sigma_log * math.sqrt(2 * math.pi)
    )
    pdf[(xs < 1) | (xs > trunc_max)] = 0.0
    pmf = pdf / (pdf.sum() + 1e-12)
    return np.log(pmf + 1e-300)


# ---------------------------------------------------------------------------
# HSMM Viterbi decoding
# ---------------------------------------------------------------------------


def _hsmm_viterbi_log(
    log_emissions: np.ndarray,
    init_logp: np.ndarray,
    logA: np.ndarray,
    dur_logpmf: list[np.ndarray],
    D_max: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Viterbi decoding for an explicit-duration HMM (HSMM).

    Parameters
    ----------
    log_emissions : (T, S) array of per-epoch log-emission probabilities.
    init_logp : (S,) initial state log-probabilities.
    logA : (S, S) log-transition matrix (row = from, col = to).
    dur_logpmf : list of S arrays, each of length D_max, giving log P(duration = d | state).
    D_max : maximum segment duration in epochs.

    Returns
    -------
    states : 1-D array of decoded state indices per segment.
    durs : 1-D array of segment durations.
    starts : 1-D array of segment start epoch indices.
    """
    T, S = log_emissions.shape

    # Cumulative sum for fast segment log-likelihood computation
    csum = np.zeros((T + 1, S), dtype=float)
    csum[1:] = np.cumsum(log_emissions, axis=0)

    dp = np.full((T, S), -np.inf, dtype=float)
    bp_state = np.full((T, S), -1, dtype=int)
    bp_dur = np.full((T, S), -1, dtype=int)

    for t in range(T):
        d_max_here = min(D_max, t + 1)
        for s in range(S):
            best_score = -np.inf
            best_prev = -1
            best_d = -1
            dur_lp = dur_logpmf[s]
            for d in range(1, d_max_here + 1):
                seg_ll = csum[t + 1, s] - csum[t + 1 - d, s]
                lp_d = dur_lp[d - 1] if d - 1 < len(dur_lp) else -np.inf
                if t - d < 0:
                    score = init_logp[s] + seg_ll + lp_d
                else:
                    prev_scores = dp[t - d, :] + logA[:, s]
                    q = int(np.argmax(prev_scores))
                    score = prev_scores[q] + seg_ll + lp_d
                if score > best_score:
                    best_score = score
                    best_prev = q if t - d >= 0 else -1
                    best_d = d
            dp[t, s] = best_score
            bp_state[t, s] = best_prev
            bp_dur[t, s] = best_d

    # Backtrace
    t = T - 1
    s = int(np.argmax(dp[t, :]))
    states_list, durs_list, starts_list = [], [], []
    while t >= 0 and s >= 0:
        d = int(bp_dur[t, s])
        start = t - d + 1
        states_list.append(s)
        durs_list.append(d)
        starts_list.append(start)
        s = int(bp_state[t, s])
        t = start - 1
        if t < 0:
            break

    return (
        np.array(states_list[::-1], dtype=int),
        np.array(durs_list[::-1], dtype=int),
        np.array(starts_list[::-1], dtype=int),
    )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def build_hsmm_components(
    hsmm_cfg: HSMMConfig,
    epoch_len_s: float,
    state_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], int]:
    """Build all HSMM components from configuration.

    Parameters
    ----------
    hsmm_cfg : HSMM configuration.
    epoch_len_s : epoch length in seconds.
    state_names : state names in canonical order.

    Returns
    -------
    init_logp : (S,) initial log-probs.
    logA : (S, S) log-transition matrix.
    dur_logpmf : list of S arrays for duration log-PMFs.
    D_max : maximum segment duration in epochs.
    """
    if state_names is None:
        state_names = STATE_NAMES
    S = len(state_names)

    # Transition matrix: small base probability for transitions, high for self-loops
    logA = np.full((S, S), math.log(1e-6), dtype=float)
    for s in range(S):
        for sp in range(S):
            if s == sp:
                logA[s, sp] = math.log(0.90) + hsmm_cfg.stay_bonus
            else:
                logA[s, sp] = math.log(0.05) + hsmm_cfg.change_penalty

    # Normalize rows
    row_logsum = np.logaddexp.reduce(logA, axis=1)
    logA = logA - row_logsum[:, None]

    # Uniform initial
    init_logp = np.full(S, -math.log(S))

    # Duration distributions
    D_max = int(max(1, math.floor(hsmm_cfg.max_dur_s / epoch_len_s)))
    d_range = np.arange(1, D_max + 1, dtype=int)
    dur_logpmf = []
    for name in state_names:
        mean_epochs = hsmm_cfg.mean_dur_s.get(name, 30.0) / epoch_len_s
        dur_logpmf.append(
            _discrete_trunc_lognorm_logpmf(d_range, mean_epochs, hsmm_cfg.lognorm_sigma, D_max)
        )

    return init_logp, logA, dur_logpmf, D_max


def decode_session(
    log_probs: np.ndarray,
    hsmm_cfg: HSMMConfig,
    epoch_len_s: float,
    state_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run HSMM Viterbi decoding on per-epoch log-probabilities for one session.

    Parameters
    ----------
    log_probs : (T, S) log-emission probabilities (e.g., from classifier).
    hsmm_cfg : HSMM configuration.
    epoch_len_s : epoch length in seconds.
    state_names : state names in canonical order.

    Returns
    -------
    states : 1-D array of decoded state indices per segment.
    durs : 1-D array of segment durations.
    starts : 1-D array of segment start epoch indices.
    """
    init_logp, logA, dur_logpmf, D_max = build_hsmm_components(
        hsmm_cfg, epoch_len_s, state_names,
    )
    return _hsmm_viterbi_log(log_probs, init_logp, logA, dur_logpmf, D_max)

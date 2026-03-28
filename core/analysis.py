from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.linalg import subspace_angles


# ============================================================
# Utilities
# ============================================================

def _sorted_snapshots(
    snapshots: List[Dict],
    phase: Optional[str] = None,
    steps: Optional[Sequence[int]] = None,
    mode: Optional[str] = "train",
) -> List[Dict]:
    selected = snapshots
    if mode is not None:
        if mode not in {"train", "eval"}:
            raise ValueError(f"mode must be 'train' or 'eval', got {mode}")
        selected = [s for s in selected if s["mode"] == mode]
    if phase is not None:
        selected = [s for s in selected if s["phase"] == phase]
    if steps is not None:
        step_set = set(steps)
        selected = [s for s in selected if s["step"] in step_set]
    if len(selected) == 0:
        raise ValueError(f"No snapshots found for phase={phase}, steps={steps}")
    return sorted(selected, key=lambda s: s["step"])


def _select_states(states: np.ndarray, snap: Dict, use_bci_only: bool) -> np.ndarray:
    """
    states: [time, n_rec] -> [time, n_sel]
    """
    if use_bci_only:
        idx = snap["bci_indices"].numpy()
        return states[:, idx]
    return states


def _select_axis(axis: np.ndarray, snap: Dict, use_bci_only: bool) -> np.ndarray:
    """
    axis: [n_rec] -> [n_sel]
    """
    if use_bci_only:
        idx = snap["bci_indices"].numpy()
        return axis[idx]
    return axis


def build_trial_tensor(
    snapshots: List[Dict],
    phase: Optional[str] = None,
    steps: Optional[Sequence[int]] = None,
    mode: Optional[str] = "train",
    use_bci_only: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    """
    Returns
    -------
    X : [n_trials, time, n_neurons]
    A : [n_trials, n_neurons]
    epoch_ids : [time]
    snaps : sorted snapshots
    """
    snaps = _sorted_snapshots(snapshots, phase=phase, steps=steps, mode=mode)

    X, A = [], []
    for s in snaps:
        states = s["states"].numpy()
        axis = s["axis"].numpy()

        X.append(_select_states(states, s, use_bci_only=use_bci_only))
        A.append(_select_axis(axis, s, use_bci_only=use_bci_only))

    X = np.stack(X, axis=0)
    A = np.stack(A, axis=0)
    epoch_ids = snaps[0]["epoch_ids"].numpy()
    return X, A, epoch_ids, snaps


def get_last_k_time_mask(T: int, last_k_time: int) -> np.ndarray:
    if last_k_time <= 0 or last_k_time > T:
        raise ValueError(f"last_k_time must be in [1, T], got {last_k_time}, T={T}")
    mask = np.zeros(T, dtype=bool)
    mask[T - last_k_time:] = True
    return mask


def make_backward_windows(
    n_trials: int,
    window_size_trials: int,
    stride_trials: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """
    Backward-looking windows ending at each trial T.

    Returns windows in python slice convention [start:end),
    where end is the current trial index + 1.

    Example:
      if window_size_trials = 10 and end = 15,
      window = [5:15] -> trials 5..14
    """
    if stride_trials is None:
        stride_trials = window_size_trials
    if window_size_trials > n_trials:
        raise ValueError(f"window_size_trials={window_size_trials} > n_trials={n_trials}")

    windows = []
    for end in range(window_size_trials, n_trials + 1, stride_trials):
        start = end - window_size_trials
        windows.append((start, end))
    return windows


# ============================================================
# PCA / dimensionality / variance
# ============================================================

def fit_pca_safe(X_flat: np.ndarray, n_components: Optional[int] = None) -> PCA:
    n_samples, n_features = X_flat.shape
    max_rank = min(n_samples, n_features)

    if n_components is None:
        pca = PCA()
    else:
        n_components = min(n_components, max_rank)
        pca = PCA(n_components=n_components)

    pca.fit(X_flat)
    return pca


def effective_dimensionality_95(X_flat: np.ndarray) -> Tuple[int, PCA]:
    """
    Smallest number of PCs explaining >=95% variance.
    """
    pca = fit_pca_safe(X_flat, n_components=None)
    csum = np.cumsum(pca.explained_variance_ratio_)
    d95 = int(np.searchsorted(csum, 0.95) + 1)
    return d95, pca


def trajectory_variance_ambient(
    X_window: np.ndarray,
) -> float:
    """
    Across-trial variance at matched times, summed over neurons,
    averaged over selected time points.

    X_window: [n_trials, time, n_neurons]
    """
    var_tn = np.var(X_window, axis=0, ddof=1)   # [t_sel, n_neurons]
    total_var_t = np.sum(var_tn, axis=1)        # [t_sel]
    return float(np.mean(total_var_t))


def trajectory_variance_pca(
    X_window: np.ndarray,
    pca: PCA,
) -> float:
    """
    Same as ambient variance, but after projection into PCA space.
    """
    n_trials, t_sel, n_neurons = X_window.shape

    X_proj = pca.transform(X_window.reshape(n_trials * t_sel, n_neurons))
    X_proj = X_proj.reshape(n_trials, t_sel, -1)

    var_tm = np.var(X_proj, axis=0, ddof=1)
    total_var_t = np.sum(var_tm, axis=1)
    return float(np.mean(total_var_t))


def manifold_basis_from_pca(pca: PCA, m: int) -> np.ndarray:
    m = min(m, pca.components_.shape[0])
    return pca.components_[:m].T


def principal_angle_summary(U1: np.ndarray, U2: np.ndarray) -> Dict[str, float]:
    angles_rad = subspace_angles(U1, U2)
    angles_deg = np.degrees(angles_rad)
    return {
        "min_angle_deg": float(np.min(angles_deg)),
        "max_angle_deg": float(np.max(angles_deg)),
        "mean_angle_deg": float(np.mean(angles_deg)),
    }


# ============================================================
# Trajectory direction / decoder alignment
# ============================================================

def mean_trajectory_direction(
    X_window: np.ndarray,
) -> np.ndarray:
    """
    Mean endpoint displacement direction over selected time interval.

    X_window: [n_trials, time, n_neurons]
    Returns unit vector [n_neurons]
    """
    start = X_window[:, 0, :]
    end = X_window[:, -1, :]
    disp = end - start                     # [n_trials, n_neurons]
    mean_disp = disp.mean(axis=0)

    norm = np.linalg.norm(mean_disp)
    if norm < 1e-12:
        return np.zeros_like(mean_disp)
    return mean_disp / norm


def decoder_alignment_with_mean_direction(
    axis_window: np.ndarray,
    mean_dir: np.ndarray,
) -> Dict[str, float]:
    """
    axis_window: [n_trials, n_neurons]
    mean_dir: [n_neurons]
    Compare the mean decoder axis in the window to the mean trajectory direction.
    """
    axis_mean = axis_window.mean(axis=0)
    axis_norm = np.linalg.norm(axis_mean)
    dir_norm = np.linalg.norm(mean_dir)

    if axis_norm < 1e-12 or dir_norm < 1e-12:
        return {
            "cosine_alignment": np.nan,
            "angle_deg": np.nan,
        }

    a = axis_mean / axis_norm
    d = mean_dir / dir_norm
    cos = float(np.clip(np.dot(a, d), -1.0, 1.0))
    ang = float(np.degrees(np.arccos(cos)))

    return {
        "cosine_alignment": cos,
        "angle_deg": ang,
    }


def trajectory_point_alignment_pca(
    X_window: np.ndarray,
    axis_window: np.ndarray,
    pca: PCA,
) -> Dict[str, float]:
    """
    For each trajectory, compute alignment as mean cosine similarity 
    between each point along the trajectory and the mean decoder axis,
    all computed in PCA space.
    
    X_window: [n_trials, time, n_neurons]
    axis_window: [n_trials, n_neurons]
    pca: fitted PCA model
    
    Returns dict with per-trial and aggregate alignment metrics.
    """
    n_trials, T, n_neurons = X_window.shape
    
    # Project trajectories to PCA space
    X_flat = X_window.reshape(n_trials * T, n_neurons)
    X_proj = pca.transform(X_flat).reshape(n_trials, T, -1)
    
    # Project axes to PCA space
    axis_proj = axis_window @ pca.components_.T  # [n_trials, n_pca]
    
    # Mean axis in PCA space
    axis_mean = axis_proj.mean(axis=0)
    axis_mean_norm = np.linalg.norm(axis_mean)
    
    if axis_mean_norm < 1e-12:
        return {
            "mean_alignment_cosine": np.nan,
            "std_alignment_cosine": np.nan,
            "per_trial_alignment": np.full(n_trials, np.nan),
        }
    
    axis_mean_normalized = axis_mean / axis_mean_norm
    
    # mean trajectory direction in PCA space
    mean_dir = mean_trajectory_direction(X_window)  # [n_neurons]
    mean_dir_proj = mean_dir @ pca.components_.T  # [n_pca]
    mean_dir_proj_norm = np.linalg.norm(mean_dir_proj)
    if mean_dir_proj_norm < 1e-12:
        return {
            "mean_alignment_cosine": np.nan,
            "std_alignment_cosine": np.nan,
            "per_trial_alignment": np.full(n_trials, np.nan),
        }
    mean_dir_proj_normalized = mean_dir_proj / mean_dir_proj_norm

    
    # Compute alignment for each trajectory
    per_trial_cosines = []
    for trial_idx in range(n_trials):
        traj = X_proj[trial_idx]  # [T, n_pca]
        axis_trial = axis_proj[trial_idx]  # [n_pca]
        
        # Cosine similarity between each point and mean axis
        norms = np.linalg.norm(traj, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        traj_normalized = traj / norms

        axis_trial_norm = np.linalg.norm(axis_trial)
        if axis_trial_norm < 1e-12:
            per_trial_cosines.append(np.nan)
            continue
        axis_trial_normalized = axis_trial / axis_trial_norm

        cosines = np.dot(traj_normalized, axis_mean_normalized)
        mean_cosine = float(np.mean(cosines))
        per_trial_cosines.append(mean_cosine)
    
    per_trial_cosines = np.array(per_trial_cosines)
    
    return {
        "mean_alignment_cosine": float(np.nanmean(per_trial_cosines)),
        "std_alignment_cosine": float(np.nanstd(per_trial_cosines)),
        "per_trial_alignment": per_trial_cosines,
    }

# ============================================================
# Endpoint cloud analysis
# ============================================================

def endpoint_cloud_last_k(
    X_window: np.ndarray,
    time_mask: np.ndarray,
) -> np.ndarray:
    """
    Average over the last-k time points to define one endpoint per trial.

    Returns [n_trials, n_neurons]
    """
    X_sel = X_window[:, time_mask, :]
    endpoints = X_sel.mean(axis=1)
    return endpoints


def fit_endpoint_pca(
    endpoints: np.ndarray,
    n_components: int = 3,
) -> Tuple[PCA, np.ndarray]:
    pca = fit_pca_safe(endpoints, n_components=n_components)
    endpoints_pca = pca.transform(endpoints)
    return pca, endpoints_pca


# ============================================================
# Main windowed geometry analysis
# ============================================================

def analyze_geometry_windows(
    snapshots: List[Dict],
    phase: Optional[str] = None,
    mode: Optional[str] = "train",
    steps: Optional[Sequence[int]] = None,
    use_bci_only: bool = True,
    window_size_trials: int = 10,
    stride_trials: Optional[int] = None,
    pca_dim_m: int = 10,
) -> Dict[str, object]:
    """
    Backward-looking window analysis:
    each window uses the previous k trials up to current trial T.
    """
    X, A, epoch_ids, snaps = build_trial_tensor(
        snapshots=snapshots,
        phase=phase,
        steps=steps,
        use_bci_only=use_bci_only,
        mode=mode,
    )

    n_trials, T, n_neurons = X.shape
    windows = make_backward_windows(
        n_trials=n_trials,
        window_size_trials=window_size_trials,
        stride_trials=stride_trials,
    )

    results = {
        "X": X,
        "A": A,
        "epoch_ids": epoch_ids,
        "selected_snapshots": snaps,
        "windows": windows,
        "window_metrics": [],
        "consecutive_alignment": [],
        "window_bases": [],
        "window_pcas": [],
    }

    bases = []

    for w_idx, (start, end) in enumerate(windows):
        Xw = X[start:end]                    # [k_trials, T, n_neurons]
        Aw = A[start:end]                    # [k_trials, n_neurons]
        Xw_flat = Xw.reshape(-1, n_neurons)

        # dimensionality on this trial window
        d95, pca_full = effective_dimensionality_95(Xw_flat)

        # manifold from top-m PCA
        pca_m = fit_pca_safe(Xw_flat, n_components=pca_dim_m)
        U = manifold_basis_from_pca(pca_m, pca_dim_m)

        # trajectory variance
        var_ambient = trajectory_variance_ambient(Xw)
        var_pca = trajectory_variance_pca(Xw, pca=pca_m)

        # decoder alignment with mean trajectory direction
        mean_dir = mean_trajectory_direction(Xw)
        align = decoder_alignment_with_mean_direction(Aw, mean_dir)

        
        align_pca = trajectory_point_alignment_pca(Xw, Aw, pca=pca_m)

        metric = {
            "window_index": w_idx,
            "start_trial_idx": start,
            "end_trial_idx": end - 1,
            "start_step": snaps[start]["step"],
            "end_step": snaps[end - 1]["step"],
            "effective_dim_95": d95,
            "trajectory_variance_ambient": var_ambient,
            "trajectory_variance_pca": var_pca,
            "decoder_alignment_pointwise_cosine": align_pca["mean_alignment_cosine"],
            "decoder_alignment_mean_angle_deg": align["angle_deg"],
        }

        results["window_metrics"].append(metric)
        results["window_bases"].append(U)
        results["window_pcas"].append(pca_m)
        bases.append(U)

    for i in range(len(bases) - 1):
        ang = principal_angle_summary(bases[i], bases[i + 1])
        ang.update({
            "window_index_1": i,
            "window_index_2": i + 1,
            "start_step_1": results["window_metrics"][i]["start_step"],
            "end_step_1": results["window_metrics"][i]["end_step"],
            "start_step_2": results["window_metrics"][i + 1]["start_step"],
            "end_step_2": results["window_metrics"][i + 1]["end_step"],
        })
        results["consecutive_alignment"].append(ang)

    return results


# ============================================================
# Global PCA over all trials
# ============================================================

def fit_global_pca(
    snapshots: List[Dict],
    phase: Optional[str] = None,
    mode: Optional[str] = "train",
    steps: Optional[Sequence[int]] = None,
    use_bci_only: bool = True,
    n_components: int = 3,
) -> Dict[str, object]:
    X, A, epoch_ids, snaps = build_trial_tensor(
        snapshots=snapshots,
        phase=phase,
        steps=steps,
        use_bci_only=use_bci_only,
        mode=mode,
    )

    n_trials, T, n_neurons = X.shape
    X_flat = X.reshape(n_trials * T, n_neurons)
    pca = fit_pca_safe(X_flat, n_components=n_components)

    X_pca = pca.transform(X_flat).reshape(n_trials, T, -1)
    A_pca = A @ pca.components_.T

    return {
        "pca": pca,
        "X": X,
        "A": A,
        "X_pca": X_pca,
        "A_pca": A_pca,
        "epoch_ids": epoch_ids,
        "selected_snapshots": snaps,
        "explained_variance_ratio": pca.explained_variance_ratio_,
    }


# ============================================================
# Per-phase PCA (3 PCs) and endpoint clouds
# ============================================================

def fit_phase_pca(
    snapshots: List[Dict],
    phase: str,
    mode: Optional[str] = "train",
    use_bci_only: bool = True,
    n_components: int = 3,
) -> Dict[str, object]:
    return fit_global_pca(
        snapshots=snapshots,
        phase=phase,
        use_bci_only=use_bci_only,
        n_components=n_components,
        mode=mode,
    )


def fit_phase_endpoint_cloud_pca(
    snapshots: List[Dict],
    phase: str,
    mode: Optional[str] = "train",
    use_bci_only: bool = True,
    last_k_time: int = 20,
    n_components: int = 3,
) -> Dict[str, object]:
    X, A, epoch_ids, snaps = build_trial_tensor(
        snapshots=snapshots,
        phase=phase,
        use_bci_only=use_bci_only,
        mode=mode,
    )

    n_trials, T, n_neurons = X.shape
    time_mask = get_last_k_time_mask(T, last_k_time)
    endpoints = endpoint_cloud_last_k(X, time_mask=time_mask)   # [n_trials, n_neurons]

    pca, endpoints_pca = fit_endpoint_pca(endpoints, n_components=n_components)

    return {
        "pca": pca,
        "endpoints": endpoints,
        "endpoints_pca": endpoints_pca,
        "selected_snapshots": snaps,
        "explained_variance_ratio": pca.explained_variance_ratio_,
    }


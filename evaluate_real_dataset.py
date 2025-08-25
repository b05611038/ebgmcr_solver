import os
import copy
import argparse

import torch
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from typing import Tuple, Optional, List, Dict

from ebgmcr import DenseEBgMCR 
from ebgmcr.eval import StreamingMetrics 
from run_real_ebgmcr import prepare_dataset

def _safe_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return torch.clamp(x.norm(dim=dim), min=eps)

def _safe_dot(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return (a * b).sum(dim=dim)

# =========================
# Hungarian (linear assignment) in PyTorch
# Minimizes cost; set maximize=True to maximize a score by negating
# =========================
def hungarian(cost: torch.Tensor, maximize: bool = False) -> List[int]:
    """
    Hungarian algorithm (O(n^3)) implemented in PyTorch.
    Args:
        cost: (n, n) cost matrix (float32/float64). Will be copied to CPU.
        maximize: if True, solves argmax by minimizing -cost.
    Returns:
        perm: list of length n with assignment j = perm[i] (column for each row).
    """
    C = cost.detach().cpu().clone()
    if maximize:
        C = -C
    n = C.size(0)
    assert C.size(1) == n, "Hungarian requires a square matrix."

    u = torch.zeros(n + 1, dtype=C.dtype)
    v = torch.zeros(n + 1, dtype=C.dtype)
    p = torch.zeros(n + 1, dtype=torch.long)
    way = torch.zeros(n + 1, dtype=torch.long)

    for i in range(1, n + 1):
        p[0] = i
        j0 = torch.tensor(0, dtype=torch.long)
        minv = torch.full((n + 1,), float('inf'), dtype=C.dtype)
        used = torch.zeros(n + 1, dtype=torch.bool)
        while True:
            used[j0] = True
            i0 = p[j0].item()
            # compute reduced costs for all unused columns
            cur = C[i0 - 1] - u[i0] - v[1:]
            cur0 = torch.cat([torch.tensor([float('inf')], dtype=C.dtype), cur], dim=0)  # align to 1..n
            # relax edges
            delta = float('inf')
            j1 = torch.tensor(0, dtype=torch.long)
            for j in range(1, n + 1):
                if not used[j]:
                    if cur0[j] < minv[j]:
                        minv[j] = cur0[j]
                        way[j] = j0
                    if minv[j].item() < delta:
                        delta = minv[j].item()
                        j1 = torch.tensor(j)
            # update potentials
            for j in range(0, n + 1):
                if used[j]:
                    u[p[j]] = u[p[j]] + delta
                    v[j] = v[j] - delta
                else:
                    minv[j] = minv[j] - delta
            j0 = j1
            if p[j0] == 0:
                break
        # augmenting
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    # p[j] = i means row i assigned to column j
    perm = [None] * n
    for j in range(1, n + 1):
        i = p[j].item()
        perm[i - 1] = j - 1
    return perm

# =========================
# Similarities & Matching
# =========================
def cosine_similarity_matrix(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Cosine similarity between rows of A and B.
    Args:
        A: (n, d)
        B: (m, d)
    Returns:
        sim: (n, m) with cos(A_i, B_j)
    """
    A_norm = A / _safe_norm(A, dim=1, eps=eps).unsqueeze(1)
    B_norm = B / _safe_norm(B, dim=1, eps=eps).unsqueeze(1)
    return A_norm @ B_norm.t()

def correlation_matrix(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Pearson correlation between columns of A and B.
    Args:
        A: (n_samples, p)  -> p variables/components (columns)
        B: (n_samples, q)  -> q variables/components (columns)
    Returns:
        corr: (p, q)
    """
    A0 = A - A.mean(dim=0, keepdim=True)
    B0 = B - B.mean(dim=0, keepdim=True)
    num = A0.t() @ B0
    den = (_safe_norm(A0, dim=0, eps=eps).unsqueeze(1) * _safe_norm(B0, dim=0, eps=eps).unsqueeze(0))
    return num / torch.clamp(den, min=eps)

def match_by_spectral_cosine(S_true: torch.Tensor, S_hat: torch.Tensor) -> Tuple[List[int], torch.Tensor]:
    """
    Match components via spectral cosine similarity (rows = components).
    Args:
        S_true: (k_true, L)
        S_hat:  (k_hat, L); assumes k_hat == k_true for matching; if not, uses min(k_true, k_hat) square subproblem.
    Returns:
        perm: list with length k (perm[i]=j)
        sims: per-pair cosine similarity tensor (k,)
    """
    k = min(S_true.size(0), S_hat.size(0))
    sim = cosine_similarity_matrix(S_true[:k], S_hat[:k])  # (k,k)
    perm = hungarian(-sim, maximize=False)  # maximize similarity == minimize -sim
    sims = sim[torch.arange(k), torch.tensor(perm)]
    return perm, sims

def match_by_concentration_corr(C_true: torch.Tensor, C_hat: torch.Tensor) -> Tuple[List[int], torch.Tensor]:
    """
    Match components via concentration correlation (columns = components).
    Args:
        C_true: (m, k_true)
        C_hat:  (m, k_hat)
    Returns:
        perm: list length k
        cors: per-pair |corr| tensor (k,)
    """
    k = min(C_true.size(1), C_hat.size(1))
    corr = correlation_matrix(C_true[:, :k], C_hat[:, :k]).abs()  # robust to sign
    perm = hungarian(-corr, maximize=False)
    cors = corr[torch.arange(k), torch.tensor(perm)]
    return perm, cors

# =========================
# Positive least-squares scalars (per component)
# =========================
def positive_ls_scale(pred: torch.Tensor, ref: torch.Tensor, dim: int = 0, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute alpha >= 0 minimizing ||ref - alpha * pred||^2 along 'dim'.
    Shapes:
        pred, ref: (..., n) aligned along 'dim'
    Returns:
        alpha: broadcastable scalar per leading index (i.e., per component)
    """
    num = _safe_dot(pred, ref, dim=dim)
    den = torch.clamp(_safe_dot(pred, pred, dim=dim), min=eps)
    return torch.clamp(num / den, min=0.0)

# =========================
# Concentration Metrics
# =========================
def rmse_per_component(C_true: torch.Tensor, C_hat: torch.Tensor) -> torch.Tensor:
    m = C_true.size(0)
    return torch.sqrt(torch.clamp(((C_true - C_hat) ** 2).sum(dim=0) / m, min=0.0))

def r2_per_component(C_true: torch.Tensor, C_hat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    resid = ((C_true - C_hat) ** 2).sum(dim=0)
    denom = ((C_true - C_true.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
    return 1.0 - resid / torch.clamp(denom, min=eps)

def nmse_per_component(C_true: torch.Tensor, C_hat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    num = ((C_true - C_hat) ** 2).sum(dim=0)
    den = torch.clamp((C_true ** 2).sum(dim=0), min=eps)
    return num / den

# =========================
# Spectral Metrics (carbs only)
# =========================
def cosine_per_component(S_true: torch.Tensor, S_hat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    a = S_true / _safe_norm(S_true, dim=1, eps=eps).unsqueeze(1)
    b = S_hat  / _safe_norm(S_hat,  dim=1, eps=eps).unsqueeze(1)
    return torch.clamp((a * b).sum(dim=1), min=-1.0, max=1.0)

def spectral_rmse_per_component(S_true: torch.Tensor, S_hat: torch.Tensor) -> torch.Tensor:
    L = S_true.size(1)
    return torch.sqrt(torch.clamp(((S_true - S_hat) ** 2).sum(dim=1) / L, min=0.0))

def negativity_penalty(S_hat: torch.Tensor, reduce: str = "sum") -> torch.Tensor:
    pen = torch.relu(-S_hat).sum()
    if reduce == "mean":
        return pen / S_hat.numel()
    return pen

# =========================
# Alignment + Rescaling Wrappers
# =========================
def align_by_spectra_then_rescale(
    S_true_tr: torch.Tensor, S_hat_tr: torch.Tensor,
    C_true_tr: Optional[torch.Tensor] = None, C_hat_tr: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Use TRAIN spectra to get permutation; compute per-component positive scales for S and (optionally) C.
    Returns:
        dict with keys: 'perm' (LongTensor shape (k,)),
                        'beta' (spectral scale per comp, (k,)),
                        'alpha' (concentration scale per comp, (k,)) if C provided
    """
    perm, _ = match_by_spectral_cosine(S_true_tr, S_hat_tr)
    k = len(perm)
    idx = torch.tensor(perm, dtype=torch.long)

    # Spectral scales beta (per comp, >=0) on train
    beta = positive_ls_scale(S_hat_tr[idx], S_true_tr[:k], dim=1)

    out = {"perm": idx, "beta": beta}
    if (C_true_tr is not None) and (C_hat_tr is not None):
        alpha = positive_ls_scale(C_hat_tr[:, idx], C_true_tr[:, :k], dim=0)
        out["alpha"] = alpha

    return out

def align_by_conc_then_rescale(
    C_true_tr: torch.Tensor, C_hat_tr: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Use TRAIN concentrations to get permutation & positive scales alpha.
    Returns:
        dict with keys: 'perm' (LongTensor (k,)), 'alpha' (k,)
    """
    perm, _ = match_by_concentration_corr(C_true_tr, C_hat_tr)
    idx = torch.tensor(perm, dtype=torch.long)
    alpha = positive_ls_scale(C_hat_tr[:, idx], C_true_tr[:, :len(perm)], dim=0)
    return {"perm": idx, "alpha": alpha}

# =========================
# Apply alignment/scales & Score
# =========================
def apply_perm_scale_C(C_hat: torch.Tensor, perm: torch.Tensor, alpha: Optional[torch.Tensor]) -> torch.Tensor:
    X = C_hat[:, perm]
    if alpha is not None:
        X = X * alpha  # broadcast over rows

    return X

def apply_perm_scale_S(S_hat: torch.Tensor, perm: torch.Tensor, beta: Optional[torch.Tensor]) -> torch.Tensor:
    X = S_hat[perm, :]
    if beta is not None:
        X = X * beta.unsqueeze(1)
    return X

def score_concentrations(
    C_true: torch.Tensor,                 # (m, K_true)
    C_hat_aligned: torch.Tensor,          # (m, K_match) after APPLYING train-time perm & alpha
    true_indices: Optional[torch.Tensor] = None,  # (K_match,) indices into true columns; if None, assumes 0..K_match-1
    policy: str = "min_k",                # "min_k" | "padded"
    fill_value: float = float("nan"),     # used only when policy="padded"
) -> Dict[str, torch.Tensor]:
    """
    Robust, CV-safe concentration scoring with K_hat != K_true.

    Parameters
    ----------
    C_true : (m, K_true)
        Ground-truth concentrations.
    C_hat_aligned : (m, K_match)
        Estimated concentrations **already permuted and scaled** to match the selected true components,
        in the SAME order as `true_indices`.
        (Typically produced by: Hungarian on TRAIN fold -> select subset -> order by true-comp id -> scale by alpha.)
    true_indices : (K_match,) or None
        Which true components are being compared. If None, we assume they are [0..K_match-1].
    policy : {"min_k","padded"}
        - "min_k": return per-component vectors of length K_match.
        - "padded": return length K_true, with unmatched true components filled with `fill_value`.
    fill_value : float
        Used only when policy="padded".

    Returns
    -------
    dict with:
      "rmse", "r2", "nmse" : per-component tensors (length K_match for min_k, length K_true for padded)
      "k_true", "k_hat_input", "matched_count", "coverage" : scalars
    """
    assert C_true.dim() == 2 and C_hat_aligned.dim() == 2, "C_true and C_hat_aligned must be 2D (m, K)."
    m, K_true = C_true.shape
    mh, K_match = C_hat_aligned.shape
    assert m == mh, "Sample dimension (m) must match between C_true and C_hat_aligned."

    # If caller didn't provide true_indices, assume we are comparing to the first K_match true columns.
    if true_indices is None:
        K_assumed = min(K_true, K_match)
        true_indices = torch.arange(K_assumed, device=C_true.device)
        # If K_match > K_true (shouldn’t happen after alignment), truncate:
        if K_match != K_assumed:
            C_hat_aligned = C_hat_aligned[:, :K_assumed]
            K_match = K_assumed

    # Sanity checks on indices
    assert true_indices.dim() == 1 and true_indices.numel() == K_match, \
        "true_indices must have length equal to number of columns in C_hat_aligned."
    assert int(true_indices.max().item()) < K_true and int(true_indices.min().item()) >= 0, \
        "true_indices out of bounds."

    # Slice the TRUE columns that correspond to the matched estimated columns
    C_true_matched = C_true[:, true_indices]

    # Compute per-component metrics on the matched subset
    rmse_k  = rmse_per_component(C_true_matched, C_hat_aligned)
    r2_k    = r2_per_component(C_true_matched, C_hat_aligned)
    nmse_k  = nmse_per_component(C_true_matched, C_hat_aligned)

    # Shape & policy handling
    if policy == "min_k":
        rmse_out = rmse_k
        r2_out   = r2_k
        nmse_out = nmse_k
    elif policy == "padded":
        # Produce length-K_true vectors; place scores at true_indices, fill others
        rmse_out = torch.full((K_true,), fill_value, dtype=rmse_k.dtype, device=rmse_k.device)
        r2_out   = torch.full((K_true,), fill_value, dtype=r2_k.dtype,   device=r2_k.device)
        nmse_out = torch.full((K_true,), fill_value, dtype=nmse_k.dtype, device=nmse_k.device)
        rmse_out[true_indices] = rmse_k
        r2_out[true_indices]   = r2_k
        nmse_out[true_indices] = nmse_k
    else:
        raise ValueError("policy must be 'min_k' or 'padded'.")

    return {
        "rmse": rmse_out,
        "r2":   r2_out,
        "nmse": nmse_out,
        "k_true": torch.tensor(K_true),
        "k_hat_input": torch.tensor(K_match),      # columns actually evaluated (post subselect)
        "matched_count": torch.tensor(K_match),
        "coverage": torch.tensor(float(K_match) / float(K_true) if K_true > 0 else 0.0),
    }

def score_spectra(
    S_true: torch.Tensor,             # (K_true, L)
    S_hat_aligned: torch.Tensor,      # (K_hat, L) already permuted to match best pairs
    policy: str = "min_k",            # "min_k" | "padded"
    fill_value: float = float("nan"), # used only if policy="padded"
) -> Dict[str, torch.Tensor]:
    """
    Robust spectral scoring when component counts differ.

    Returns:
      {
        "cosine": Tensor,              # shape (k) for min_k, or (K_true) for padded
        "rmse": Tensor,                # same shape as "cosine"
        "neg_penalty_total": Tensor,   # scalar (on S_hat_aligned)
        "neg_penalty_mean":  Tensor,   # scalar (on S_hat_aligned)
        "k_true": Tensor[int],         # scalar
        "k_hat":  Tensor[int],         # scalar
        "matched_count": Tensor[int],  # scalar = min(K_true, K_hat)
        "coverage": Tensor[float],     # scalar = matched_count / K_true
      }
    """
    assert S_true.dim() == 2 and S_hat_aligned.dim() == 2, "S_true and S_hat must be 2D (K, L)."
    K_true, L = S_true.shape
    K_hat,  Lh = S_hat_aligned.shape
    assert L == Lh, "Spectral length (L) must match."

    k = min(K_true, K_hat)

    # Compute pairwise metrics on the matched subset only
    cos_k  = cosine_per_component(S_true[:k, :], S_hat_aligned[:k, :])
    rmse_k = spectral_rmse_per_component(S_true[:k, :], S_hat_aligned[:k, :])

    if policy == "min_k":
        cosine_out = cos_k
        rmse_out   = rmse_k
    elif policy == "padded":
        # Create length-K_true vectors, fill first k entries with computed values, rest with fill_value
        cosine_out = torch.full((K_true,), fill_value, dtype=cos_k.dtype, device=cos_k.device)
        rmse_out   = torch.full((K_true,), fill_value, dtype=rmse_k.dtype, device=rmse_k.device)
        if k > 0:
            cosine_out[:k] = cos_k
            rmse_out[:k]   = rmse_k
    else:
        raise ValueError("policy must be 'min_k' or 'padded'.")

    # Negativity penalties depend only on the estimate (regardless of K mismatch)
    neg_total = negativity_penalty(S_hat_aligned, reduce="sum")
    neg_mean  = negativity_penalty(S_hat_aligned, reduce="mean")

    return {
        "cosine": cosine_out,
        "rmse":   rmse_out,
        "neg_penalty_total": neg_total,
        "neg_penalty_mean":  neg_mean,
        "k_true": torch.tensor(K_true),
        "k_hat":  torch.tensor(K_hat),
        "matched_count": torch.tensor(k),
        "coverage": torch.tensor(float(k) / float(K_true) if K_true > 0 else 0.0),
    }

def grab_benchmark_prediction(benchmark_path, method):
    assert method in ['Bayes-NMF', 'ICA', 'MCR-ALS', 'NMF', 'Sparse-NMF']
    assert os.path.isdir(benchmark_path)

    c_num, predictions = 1, {}
    while os.path.isdir(os.path.join(benchmark_path, f"{method}_C{c_num}")):
        valid_path = os.path.join(benchmark_path, f"{method}_C{c_num}")
        D = np.load(os.path.join(valid_path, 'reconstruction.npy'))
        C = np.load(os.path.join(valid_path, 'concentration.npy'))
        S = np.load(os.path.join(valid_path, 'component.npy'))
        predictions[c_num] = {'D': torch.from_numpy(D).to(torch.float32), 
                              'C': torch.from_numpy(C).to(torch.float32), 
                              'S': torch.from_numpy(S).to(torch.float32)}
        c_num += 1

    return predictions

def _extract_r2_from_record(csv_file):
    df = pd.read_csv(csv_file)
    r2 = df['R2'][0]
    c_num = df['effective_components'][0]
    return {'R2': r2,
            'c_num': c_num}

def grab_ebgmcr_prediction(solver_path, data, device = torch.device('cpu')):
    predictions, performance_record = {}, {}
    checkpoints = [solver_path]
    final_performance = _extract_r2_from_record(os.path.join(solver_path, 'final_result.csv'))
    for f in os.listdir(solver_path):
        if os.path.isdir(os.path.join(solver_path, f)):
            checkpoints.append(os.path.join(solver_path, f))

    for checkpoint_path in checkpoints:
        performance = None
        if os.path.isfile(os.path.join(checkpoint_path, 'result.csv')):
            performance = _extract_r2_from_record(os.path.join(checkpoint_path, 'result.csv'))
        else:
            performance = final_performance

        compute_result = False
        c_num = performance['c_num']
        if c_num not in predictions:
            compute_result = True
            performance_record[c_num] = performance['R2']
        else:
            if performance['R2'] > performance_record[c_num]:
                compute_result = True
                performance_record[c_num] = performance['R2']

        if compute_result:
            analyzer = DenseEBgMCR(num_components = 1,
                                   dim_components = 1,
                                   device = device)

            analyzer.load_mcrs(checkpoint_path)
            solver_predictions = analyzer.parse_pattern(data)
            D = solver_predictions['reconstruction']
            effective_indices = (solver_predictions['select_prob'] > analyzer.select_prob_threshld)
            effective_indices = effective_indices.sum(0) > 0
            S = solver_predictions['components'][0, effective_indices]
            C = solver_predictions['concentration'][:, effective_indices, 0]
            assert c_num == int(effective_indices.sum())

        predictions[c_num] = {'D': D, 'C': C, 'S': S}

    return predictions

def eval_solver_predictions(dataset, method_predictions):
    eval_results = {}
    if dataset['S'] is None:
        has_component = False
    else:
        has_component = True

    ground_truth_c_num = dataset['C'].shape[-1]
    for c_num in method_predictions:
        eval_results[c_num] = {}
        for method in method_predictions[c_num]:
            single_eval_result = {}
            single_prediction = method_predictions[c_num][method]
            if has_component:
                align = align_by_spectra_then_rescale(dataset['S'], 
                                                      single_prediction['S'], 
                                                      dataset['C'], 
                                                      single_prediction['C'])
            else:
                align = align_by_conc_then_rescale(dataset['C'], single_prediction['C'])

            reconstruction_metric = StreamingMetrics(dim = dataset['dim_component'])
            reconstruction_metric.update(dataset['D'], single_prediction['D'])
            reconstruct_nmse, reconstruct_r2 = reconstruction_metric.compute()

            single_eval_result['D'] = {'nmse': reconstruct_nmse,
                                       'R2': reconstruct_r2}

            aligned_C = apply_perm_scale_C(single_prediction['C'], align['perm'], align.get('alpha', None))
            concentration_eval = score_concentrations(dataset['C'], aligned_C)
            for metric_item in concentration_eval:
                if concentration_eval[metric_item].numel() > 1:
                    concentration_eval[metric_item] = concentration_eval[metric_item].mean()

                concentration_eval[metric_item] = concentration_eval[metric_item].item()

            single_eval_result['C'] = concentration_eval

            if has_component:
                aligned_S = apply_perm_scale_S(single_prediction['S'], align['perm'], align['beta'])
                component_eval = score_spectra(dataset['S'], aligned_S)
                for metric_item in component_eval:
                    if component_eval[metric_item].numel() > 1:
                        component_eval[metric_item] = component_eval[metric_item].mean()

                    component_eval[metric_item] = component_eval[metric_item].item()

                single_eval_result['S'] = component_eval
            else:
                single_eval_result['S'] = None

            eval_results[c_num][method] = single_eval_result

    metrices = {'D': ['R2', 'nmse'],
                'C': ['rmse', 'r2', 'nmse'],
                'S': ['cosine', 'rmse', 'neg_penalty_total', 'neg_penalty_mean']}

    return eval_results, metrices, ground_truth_c_num

def summarize_metric(methods, eval_item, target_metric, eval_results):
    recorded_c_nums = list(eval_results.keys())
    head_line = f'{target_metric},'
    for c_num in recorded_c_nums:
        head_line += f'{c_num},'

    head_line += '\n'
    lines = [head_line]
    for method in methods:
        single_line = f'{method},'
        for c_num in recorded_c_nums:
            value = 'N/A'
            try:
                value = eval_results[c_num][method][eval_item][target_metric]
            except KeyError:
                pass
            except TypeError:
                pass

            single_line += f'{value},'

        single_line += '\n'
        lines.append(single_line)

    lines.append('\n')

    return lines

def main():
    display_text = 'Evaluate baseline and EB-gMCR solvers on real dataset (carbs and nir). ' + \
            'Please use R run real_dataset/prepare_real_data.R to download the dataset.'
    parser = argparse.ArgumentParser(description = display_text)

    parser.add_argument('--dataset_name', type = str, required = True,
            help = 'The dataset ("carbs" and "nir") to be analyzed by EB-gMCR.')
    parser.add_argument('--ebgmcr_path', type = str, required = True,
            help = 'The directory saved by the solver training program.')
    parser.add_argument('--baseline_path', type = str, required = True,
            help = 'The directory saved by the benchmark training program.')
    parser.add_argument('--base_filename', type = str, default = 'real_result',
            help = 'The saved directory of the searched results.')

    args = parser.parse_args()
    baseline_methods = ['NMF', 'Sparse-NMF', 'Bayes-NMF', 'ICA', 'MCR-ALS']
    result_filename = f'{args.base_filename}_{args.dataset_name}.csv'
    dataset = prepare_dataset(args.dataset_name)
    method_predictions = {}
    for method in baseline_methods:
        prediction = grab_benchmark_prediction(args.baseline_path, method)
        for c_num in prediction:
            if c_num not in method_predictions:
                method_predictions[c_num] = {}

            method_predictions[c_num][method] = prediction[c_num]

    ebgmcr_prediction = grab_ebgmcr_prediction(args.ebgmcr_path, dataset['D'])
    for c_num in ebgmcr_prediction:
        if c_num not in method_predictions:
            method_predictions[c_num] = {}

        method_predictions[c_num]['EB-gMCR'] = ebgmcr_prediction[c_num]

    (eval_results,
     metrices,
     ground_truth_c_num) = eval_solver_predictions(dataset, 
                                                   method_predictions)

    all_methods = copy.deepcopy(baseline_methods)
    all_methods.append('EB-gMCR')
    lines = [f'Dataset: {args.dataset_name} (ground_truth_C_num={ground_truth_c_num})\n']
    for eval_item in metrices:
        lines.append(f'Eval_item: dataset_{eval_item}\n')
        for m in metrices[eval_item]:
            metric_lines = summarize_metric(all_methods, eval_item, m, eval_results)
            lines += metric_lines

    with open(result_filename, 'w') as F:
        F.writelines(lines)
        F.close()

    print(f'Layout complete result to file:{result_filename} done.')

    return None

if __name__ == '__main__':
    main()



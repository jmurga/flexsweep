"""
flexsweep/npe.py
================
Neural Posterior Estimation (NPE) of the joint selection coefficient ``s`` and
sweep age ``t`` from flexsweep z-scored feature vectors — the deployable Path-B
estimator, downstream of the CNN/DANN sweep classifier.

A single conditional normalizing flow (neural spline flow) is trained once on the
reference simulation table; at predict time it returns a full joint posterior
``p(s, t | fv)`` per candidate window, amortized (no per-locus simulation). A
split-conformal radius per (target, alpha) turns the raw posterior quantiles into
VALID MARGINAL (average) credible intervals; this is the standard split-conformal
guarantee under exchangeability of candidates with the reference sweeps. The
``conformal`` argument selects how that radius is computed — all deploy-time safe
(none uses the true subclass), mirroring the experiments in
``~/st_option1b/1c/1d``:

  * ``"global"``       one radius per (target, alpha)            [default, unchanged]
  * ``"normalized"``   CQR normalized by the flow's OWN posterior std (locally
                       adaptive: young/soft loci with broader posteriors get a
                       proportionally larger radius) — no strata, no labels.
  * ``"pred"``         Mondrian per PREDICTED stratum (1d): bins of predicted
                       ``t`` (young/mid/old) x predicted ``s`` (weak/strong), with a
                       global fallback for sparse strata.
  * ``"conservative"`` a single global radius inflated to the WORST predicted
                       stratum (the "1e" floor) — over-covers easy classes to
                       guarantee the hard ones.

A PCA-whitened Mahalanobis OOD gate flags candidates outside the simulated
manifold, where the flow would extrapolate.

Calibration has two distinct layers, kept separate on purpose:

  * **Intrinsic (sbi-native, diagnostic).** :func:`analyze_npe` runs sbi's own
    default calibration tools on a labeled table — Simulation-Based Calibration
    (Talts et al. 2018: ``run_sbc`` / ``check_sbc``, rank uniformity + KS + C2ST)
    and TARP expected coverage (Lemos et al. 2023: ``run_tarp`` / ``check_tarp``),
    plus sbi's ``sbc_rank_plot`` and ``pairplot``. These MEASURE whether the raw
    flow posterior is calibrated; they apply no correction. This is the analysis
    entry point going forward.
  * **Deployed (conformal, corrective).** The split-conformal radius below is the
    layer that MAKES the marginal interval valid when the intrinsic diagnostic
    shows the raw flow is (mildly) over-confident. sbi provides no such correction.

Design choices (kept identical to the validated experiments):
  * ``theta = (s, t)``. The reference parquets store ``s`` as ``-log(s)``. The
    ``s_transform`` argument selects the units the flow learns/predicts ``s`` in:
    ``"natural"`` (default) decodes to the linear selection coefficient
    (``exp(-stored)`` for base ``e``, i.e. ``-ln(s)``; ``s`` ~[0.01, 0.05]) via
    ``s_log_base``; ``"neglog"`` trains directly on the stored ``-log(s)`` as is (no
    decode, base-agnostic). Either way ``s`` is modelled directly (no extra log
    transform) and the choice is saved in the artifact, so predictions and CIs come
    back in the same units. ``t`` in generations. **Optionally add sweep POSITION**
    as a third target via
    ``targets=("s", "t", "pos")`` — ``pos`` is the relative position of the sweep in
    ``[0.1, 0.9]`` of the 1.2 Mb locus (natural scale, no transform). It is strictly
    opt-in: with the default ``targets`` the ``pos`` column is ignored even if
    present, so existing 2-D artifacts are unchanged. All targets share the one flow,
    conformal calibration, and OOD gate.
  * ``x`` = z-scored feature vector (``preprocess=False`` upstream); not re-normalized.
    The flow conditions on a learned summary from the embedding net, selected by
    ``embedding``: ``"fc"`` (default, generic MLP, structure-blind baseline);
    ``"cnn"``, sbi's built-in ``CNNEmbedding`` on the parsed grid (stats as
    ``in_channels``, auto 1D/2D — generic structured control); or ``"conv"``, the
    custom ``cnn_flexsweep`` three-branch dilated-conv tower (Conv2d for multi-scale
    fv, Conv1d for single-scale, where the 2D tower is degenerate). Three further
    sbi-native embeddings are also selectable, all wired on the same parsed grid:
    ``"causalcnn"`` (``CausalCNNEmbedding`` — a 1D dilated causal conv over the
    ``W*C`` sequence with ``S`` stat channels; the library analog of the custom
    ``conv`` dilation ladder, i.e. the fairest "does the hand-built tower beat the
    native dilated conv?" control); ``"resnet2d"`` (``ResNetEmbedding2D`` — a deeper
    residual CNN on the ``(S, W, C)`` image; needs a genuine 2-D grid, so it falls
    back to ``"fc"`` on single-scale layouts); and ``"spectralconv"``
    (``SpectralConvEmbedding`` — an FNO-style spectral conv over the sequence, suited
    to the smooth spatial/LD decay; a ``Linear`` head maps its variable-length output
    to ``emb_out``). All grid embeddings fall back to ``"fc"`` if the columns are not
    a complete ``{stat}_{window}_{center}`` grid.
  * ``condition_on=("r",)`` makes the flow learn ``q(s, t | x, r)`` — the known
    per-region recombination rate enters as a CONDITIONING channel (standardized
    ``log10``) that bypasses the embedding (ReLERNN-style side branch), NOT as a
    parameter (the prior stays 2-D over s, t). At predict the candidate parquet must
    carry the same column (the window-mean map value); breaking the s–r
    footprint-width degeneracy is expected to sharpen ``s`` most. ``("r", "mu")``
    conditions on both. The training r-prior must cover the empirical map range
    (predict flags out-of-range rows as ``cond_oob``).
  * ``delta_ihh_*`` columns are dropped (excluded everywhere in flexsweep).
  * Conditional conformal keys ONLY on deploy-available quantities (posterior
    spread, predicted s/t): the method outputs only (s, t); the subclass label is
    an evaluation grouping, never available at deploy.

Three public entry points:
  * :func:`fit_npe`     — train flow, compute global conformal q, fit OOD gate, save artifact.
  * :func:`predict_npe` — load artifact, score candidate windows -> s, t + marginal CIs + OOD flag.
  * :func:`analyze_npe` — sbi-native calibration/analysis (SBC + TARP + rank/pair plots)
                          on a labeled table; the flow's intrinsic (pre-conformal) calibration.

References
----------
Greenberg, Nonnenmacher & Macke 2019 *ICML* (APT / NPE; ``sbi`` package)
Durkan et al. 2019 *NeurIPS* (neural spline flows)
Romano, Patterson & Candès 2019 *NeurIPS* (conformalized quantile regression)
Talts, Betancourt, Simpson, Vehtari & Gelman 2018 (Simulation-Based Calibration)
Lemos, Coogan, Hezaveh & Perreault-Levasseur 2023 *ICML* (TARP expected coverage)
Min et al. 2026 (NPE for population-genetic inference; method reference)
"""

import logging
import re

import numpy as np
import polars as pl

from .fv import load_pickle, save_pickle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Non-feature columns in the reference / candidate parquet tables.
#: ``pos`` (relative sweep position in [0.1, 0.9] of the 1.2 Mb locus) is listed so
#: it is never treated as a feature; it is only USED as a target when explicitly
#: requested via ``targets=(..., "pos")`` (opt-in, see :func:`fit_npe`).
META_COLS = ("iter", "s", "t", "pos", "f_i", "f_t", "mu", "r", "model")

#: Default inference targets. ``pos`` is opt-in (append it to ``targets`` only if
#: the reference/candidate parquets carry a ``pos`` column).
DEFAULT_TARGETS = ("s", "t")

#: Credible levels for which intervals (and conformal radii) are produced.
ALPHAS = (0.50, 0.80, 0.90, 0.95)


# ---------------------------------------------------------------------------
# Lazy heavy-dependency import (torch / sbi are optional, GPU-heavy)
# ---------------------------------------------------------------------------
def _import_sbi():
    """Import torch + sbi on demand; raise a clear error if missing."""
    try:
        import torch
        try:
            from sbi.inference import NPE
        except Exception:  # older sbi naming
            from sbi.inference import SNPE as NPE
        from sbi.neural_nets import posterior_nn
        from sbi.utils import BoxUniform
        try:
            from sbi.neural_nets.embedding_nets import CNNEmbedding, FCEmbedding
        except Exception:
            from sbi.neural_nets import CNNEmbedding, FCEmbedding  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "flexsweep.npe requires `torch` and `sbi` (>=0.26). "
            "Install with `pip install sbi`."
        ) from exc
    return torch, NPE, posterior_nn, BoxUniform, FCEmbedding, CNNEmbedding


def _import_sbi_analysis():
    """Import sbi's built-in calibration diagnostics + analysis plots on demand.

    Returns ``(run_sbc, check_sbc, run_tarp, check_tarp, sbc_rank_plot, pairplot)``.
    SBC (``run_sbc``/``check_sbc``) is the hard requirement; ``run_tarp``/
    ``check_tarp`` (sbi >= 0.22) and the plot fns are ``None`` on older sbi so the
    caller degrades gracefully instead of crashing.
    """
    try:
        from sbi.diagnostics import check_sbc, run_sbc
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "flexsweep.npe.analyze_npe needs sbi's diagnostics (SBC; sbi>=0.22). "
            "Install/upgrade with `pip install -U sbi`."
        ) from exc

    def _opt(mod: str, name: str):
        try:
            return getattr(__import__(mod, fromlist=[name]), name)
        except Exception:
            return None

    return (run_sbc, check_sbc,
            _opt("sbi.diagnostics", "run_tarp"), _opt("sbi.diagnostics", "check_tarp"),
            _opt("sbi.analysis", "sbc_rank_plot"), _opt("sbi.analysis", "pairplot"))


def _to_list(v):
    """Coerce a torch tensor / ndarray / scalar diagnostic to a JSON-able list."""
    try:
        import torch
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(v).tolist()


def _posterior_device(post):
    """The device the loaded flow's parameters actually live on (fallback: cpu)."""
    import torch

    est = getattr(post, "posterior_estimator", None)
    try:
        return next(est.parameters()).device
    except Exception:
        return torch.device("cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def feature_cols(df: "pl.DataFrame") -> list[str]:
    """Feature columns: non-meta and non ``delta_ihh_*`` (excluded everywhere)."""
    return [c for c in df.columns if c not in META_COLS and "delta_ihh" not in c]


#: Supported ``s`` parametrizations for training/prediction (see :func:`fit_npe`).
S_TRANSFORMS = ("natural", "neglog")


def _s_base(base) -> float:
    """Resolve a log-base spec (``"e"``/``"ln"``/``"10"``/number) to a float."""
    if isinstance(base, str) and base.strip().lower() in ("e", "ln", "natural"):
        return float(np.e)
    return float(base)


def _decode_s(s_stored, *, transform: str = "natural", base="e") -> np.ndarray:
    """Convert the stored ``-log(s)`` column to the requested training/output units.

    The reference parquets store ``-log_base(s)``.

    * ``transform="natural"`` inverts it to the linear selection coefficient ``s``
      (``exp(-stored)`` for the natural log, ``base**(-stored)`` otherwise;
      ``s`` ~[0.01, 0.05] under the flexsweep prior).
    * ``transform="neglog"`` uses the stored ``-log(s)`` **as is** — the flow is
      trained and predicts directly in ``-log(s)`` units (base-agnostic, no decode).
    """
    x = np.asarray(s_stored, dtype=np.float64)
    if transform == "neglog":
        return x
    b = _s_base(base)
    return np.exp(-x) if b == float(np.e) else np.power(b, -x)


def _decode_target(name: str, values, *, s_transform: str = "natural",
                   s_log_base="e") -> np.ndarray:
    """Truth for a target column in the requested units.

    ``s`` is converted from the stored ``-log(s)`` per ``s_transform`` (``"natural"``
    -> linear ``s``; ``"neglog"`` -> keep ``-log(s)`` as is); ``t`` (generations) and
    ``pos`` (relative sweep position in [0.1, 0.9]) are already natural and pass
    through.
    """
    v = np.asarray(values, dtype=np.float64)
    return (_decode_s(v, transform=s_transform, base=s_log_base)
            if name == "s" else v)


def _conf_quantile(e: np.ndarray, a: float) -> float:
    """The ``⌈(n+1)a⌉``-th smallest nonconformity score (split-conformal level)."""
    n = len(e)
    if n == 0:
        return 0.0
    k = int(np.ceil((n + 1) * a))
    return float(np.max(e)) if k > n else float(np.sort(e)[k - 1])


def _cqr_q(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, a: float) -> float:
    """Global CQR conformal radius for target coverage ``a`` (Romano 2019)."""
    return _conf_quantile(np.maximum(lo - y, y - hi), a)


# ---------------------------------------------------------------------------
# Conditional conformal (selectable via ``conformal=``) — reuses the CQR score
# from the st_option1b/1c/1d experiments, all deploy-time safe (no true label).
#   global       one radius per (target, alpha)                  [default, unchanged]
#   normalized   CQR normalized by the flow's own posterior std  [adaptive, no strata]
#   pred         Mondrian per PREDICTED stratum (1d): t young/mid/old x s weak/strong
#   conservative one global radius inflated to the worst predicted stratum (1e floor)
# ---------------------------------------------------------------------------
CONFORMAL_MODES = ("global", "normalized", "pred", "conservative")


def _assign_strata(s_med, t_med, t_edges, s_cut):
    """Deployable predicted strata ``{young,mid,old}_{w,s}`` from NPE medians (1d)."""
    s_med = np.asarray(s_med); t_med = np.asarray(t_med)
    lo_t, hi_t = float(t_edges[0]), float(t_edges[1])
    tb = np.where(t_med < lo_t, "young", np.where(t_med < hi_t, "mid", "old"))
    sb = np.where(s_med < s_cut, "w", "s")
    return np.array([f"{a}_{b}" for a, b in zip(tb, sb)])


def _build_conformal(mode, dcal, yc, targets, *, t_edges, min_stratum):
    """Fit the conformal structure for ``mode`` on the calib draws.

    ``dcal`` is a list of per-locus ``(n_post, D)`` natural-unit draw arrays (column
    ``j`` = ``targets[j]``), ``yc = {name: truth[cal]}`` the calib truths. The
    predicted STRATA (pred/conservative) are keyed on the ``s`` and ``t`` medians
    only — ``pos`` (if present) simply gets its own radius within those strata.
    Returns a JSON-able dict carrying everything :func:`_conformal_radius` needs.
    """
    tidx = {nm: i for i, nm in enumerate(targets)}
    s_i, t_i = tidx.get("s"), tidx.get("t")
    s_med = np.array([np.median(d[:, s_i]) for d in dcal]) if s_i is not None else None
    t_med = np.array([np.median(d[:, t_i]) for d in dcal]) if t_i is not None else None
    sig = {nm: np.array([d[:, tidx[nm]].std() + 1e-12 for d in dcal]) for nm in targets}
    conf = {"mode": mode}

    if mode == "normalized":  # CQR score divided by the locus's own posterior std
        q = {}
        for nm in targets:
            for a in ALPHAS:
                lo, hi = _per_locus_quantiles(dcal, tidx[nm], a)
                e = np.maximum(lo - yc[nm], yc[nm] - hi) / sig[nm]
                q[f"{nm}|{a}"] = _conf_quantile(e, a)
        conf["q"] = q
        return conf

    if mode in ("pred", "conservative"):
        s_cut = float(np.median(s_med))
        strat = _assign_strata(s_med, t_med, t_edges, s_cut)
        qstrata, qfallback, qcons = {}, {}, {}
        for nm in targets:
            for a in ALPHAS:
                lo, hi = _per_locus_quantiles(dcal, tidx[nm], a)
                gq = _cqr_q(lo, hi, yc[nm], a)
                per = {}
                for v in np.unique(strat):
                    m = strat == v
                    per[v] = _cqr_q(lo[m], hi[m], yc[nm][m], a) if m.sum() >= min_stratum else gq
                key = f"{nm}|{a}"
                qstrata[key] = per
                qfallback[key] = gq
                qcons[key] = float(max([gq] + list(per.values())))  # worst-stratum floor
        if mode == "pred":
            conf.update(qstrata=qstrata, qfallback=qfallback,
                        t_edges=[float(t_edges[0]), float(t_edges[1])], s_cut=s_cut,
                        min_stratum=int(min_stratum))
        else:  # conservative: a single inflated global radius (deployable, no strata)
            conf["q"] = qcons
        return conf

    return conf  # global -> nothing extra; predict falls back to qglobal


def _conformal_radius(conf, qglobal, nm, a, n, *, s_med=None, t_med=None, sigma=None):
    """Per-locus additive radius array for (target ``nm``, level ``a``).

    ``conf`` is the artifact's conformal dict (or ``None``/global for the legacy
    path). ``loc = lo - rad``, ``hic = hi + rad``.
    """
    key = f"{nm}|{a}"
    mode = (conf or {}).get("mode", "global")
    if mode == "normalized":
        return conf["q"][key] * np.asarray(sigma)
    if mode == "pred":
        strat = _assign_strata(s_med, t_med, conf["t_edges"], conf["s_cut"])
        per, gq = conf["qstrata"][key], conf["qfallback"][key]
        return np.array([per.get(st, gq) for st in strat], dtype=float)
    # global (legacy) and conservative both carry a single scalar radius
    q = conf["q"][key] if (conf and "q" in conf) else qglobal[key]
    return np.full(int(n), float(q))


def _stratified_split(model: np.ndarray, frac: float, seed: int):
    """Return (rest_mask, picked_mask) drawing ``frac`` per model group."""
    rng = np.random.default_rng(seed)
    pick = []
    for g in np.unique(model):
        gi = np.where(model == g)[0]
        rng.shuffle(gi)
        pick.append(gi[: int(round(frac * len(gi)))])
    pick = np.concatenate(pick) if pick else np.array([], dtype=int)
    mask = np.zeros(len(model), bool)
    mask[pick] = True
    return ~mask, mask


def _sample_batch(posterior, X, n_post: int, dev: str, chunk: int = 400):
    """List of (n_post, 2) numpy draws, one per row of ``X`` (batched w/ fallback)."""
    import torch

    out = []
    for i in range(0, X.shape[0], chunk):
        xb = torch.as_tensor(X[i:i + chunk]).to(dev)
        try:
            s = posterior.sample_batched(
                (n_post,), x=xb, show_progress_bars=False
            ).cpu().numpy()
            for j in range(s.shape[1]):
                out.append(s[:, j, :])
        except Exception:
            for j in range(xb.shape[0]):
                out.append(
                    posterior.sample((n_post,), x=xb[j], show_progress_bars=False)
                    .cpu().numpy()
                )
        logger.info("sampled %d/%d", min(i + chunk, X.shape[0]), X.shape[0])
    return out


def _per_locus_quantiles(draws, idx: int, a: float):
    """Lower/upper ``a``-interval quantiles per locus for target column ``idx``.

    ``draws`` is a list of per-locus ``(n_post, D)`` arrays; column ``idx`` selects
    the target.
    """
    ql, qh = (1 - a) / 2, (1 + a) / 2
    lo = np.array([np.quantile(d[:, idx], ql) for d in draws])
    hi = np.array([np.quantile(d[:, idx], qh) for d in draws])
    return lo, hi


def _resolve_device(device: str) -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


# ---------------------------------------------------------------------------
# Structured (conv) embedding — exploit the {stat}_{window}_{center} fv grid
# ---------------------------------------------------------------------------
def _parse_grid(feat: list[str]):
    """Parse ``{stat}_{window}_{center}`` feature names into grid dims + a gather.

    Returns ``(stats, windows, center_ranks, perm)`` where ``perm`` is an index
    array such that ``X[:, perm].reshape(B, S, W, C)`` is in canonical stat-major
    order (stat, then window, then center-rank) regardless of the parquet column
    order. If the columns do not form a complete ``S*W*C`` cube, returns
    ``(None, None, None, None)`` so the caller can fall back to the FC embedding.

    Centers are ranked **within each window** (per-window ascending bp -> 0..C-1),
    not by global absolute bp. This admits two grid types with one code path:

    * Rectangular grid — every window shares the same centers, so rank ``r`` is the
      same absolute center in every window (identical to the old absolute keying).
    * Caldas multi-scale fan — each window has its OWN size-dependent centers
      (``{stat}_{window}_{center}`` with the center list fanned per window), but the
      same *count* ``d`` per window and symmetric about the sweep, so rank ``r`` is
      the same relative position across windows (rank ``(d-1)/2`` == the sweep
      center for every window). Keying on rank, not absolute bp, is what lets the
      fanned geometry form a cube — so the conv/cnn tower engages — WITHOUT renaming
      any columns (absolute-bp names are preserved for the flat consumers).

    A full ``perm`` (no ``None``) with ``S*W*C == len(feat)`` guarantees a bijection,
    so a ragged fan (unequal centers-per-window) or missing stat is rejected.
    """
    pat = re.compile(r"^(?P<stat>.+)_(?P<win>\d+)_(?P<cen>\d+)$")
    rows = []
    for i, c in enumerate(feat):
        m = pat.match(c)
        if not m:
            return None, None, None, None
        rows.append((m.group("stat"), int(m.group("win")), int(m.group("cen")), i))
    stats = sorted({r[0] for r in rows})
    wins = sorted({r[1] for r in rows})
    # Rank centers within each window; require the same count in every window.
    cens_by_win: dict[int, set] = {}
    for _stat, win, cen, _i in rows:
        cens_by_win.setdefault(win, set()).add(cen)
    counts = {len(v) for v in cens_by_win.values()}
    if len(counts) != 1:  # ragged fan across windows -> not a cube
        return None, None, None, None
    C = counts.pop()
    cen_rank = {(win, cen): r
                for win, cens in cens_by_win.items()
                for r, cen in enumerate(sorted(cens))}
    S, W = len(stats), len(wins)
    if S * W * C != len(feat):
        return None, None, None, None
    s_ix = {s: k for k, s in enumerate(stats)}
    w_ix = {w: k for k, w in enumerate(wins)}
    perm = [None] * len(feat)
    for stat, win, cen, i in rows:
        perm[(s_ix[stat] * W + w_ix[win]) * C + cen_rank[(win, cen)]] = i
    if any(p is None for p in perm):
        return None, None, None, None
    return stats, wins, list(range(C)), np.asarray(perm, dtype=np.int64)


# Embedding-net classes are defined lazily (torch is an optional heavy import) but
# registered at MODULE level so ``pickle`` can resolve them by qualname when the
# saved sbi posterior is reloaded in a fresh interpreter. ``__getattr__`` (PEP 562)
# builds them on first attribute access — which is exactly what pickle's
# ``getattr(flexsweep.npe, "FvConvEmbedding")`` triggers on load.
_EMB_CLASSES: dict = {}


def _embedding_classes() -> dict:
    """Build (once) and return the picklable embedding-net classes."""
    if _EMB_CLASSES:
        return _EMB_CLASSES

    import torch
    import torch.nn as nn

    class FvConvEmbedding(nn.Module):
        """Three-branch dilated-conv tower mirroring ``cnn_flexsweep``.

        Reshapes the flat fv into ``(batch, S stats-as-channels, W, C)`` and applies
        the tower, emitting an ``emb_out`` learned summary for the flow. **Conv2d**
        when both spatial axes are >1 (multi-scale: 3x3 dil(1,1) / 2x2 dil(1,3) /
        2x2 dil(1,5), matching ``cnn.py``); **Conv1d** with dilations {1,3,5} when
        one axis is singleton (single-scale, where the 2D tower is degenerate).
        """

        def __init__(self, S, W, C, perm, emb_out, widths, dense, dropout):
            super().__init__()
            self.S, self.W, self.C = int(S), int(W), int(C)
            self.is2d = bool(W > 1 and C > 1)
            self.register_buffer("perm", torch.as_tensor(perm, dtype=torch.long))
            c1, c2, c3 = widths
            Conv = nn.Conv2d if self.is2d else nn.Conv1d
            Pool = nn.AdaptiveAvgPool2d if self.is2d else nn.AdaptiveAvgPool1d

            def branch(k, dil):
                return nn.Sequential(
                    Conv(self.S, c1, k, padding="same", dilation=dil), nn.ReLU(),
                    Conv(c1, c2, k, padding="same", dilation=dil), nn.ReLU(),
                    Conv(c2, c3, k, padding="same", dilation=dil), nn.ReLU(),
                    Pool(1), nn.Flatten(), nn.Dropout(dropout),
                )

            if self.is2d:  # multi-scale grid: dilate across centers (axis -1)
                self.b1, self.b2, self.b3 = (branch((3, 3), (1, 1)),
                                             branch((2, 2), (1, 3)),
                                             branch((2, 2), (1, 5)))
            else:          # single-scale: 1D dilation ladder over the long axis
                self.b1, self.b2, self.b3 = branch(3, 1), branch(2, 3), branch(2, 5)
            self.head = nn.Sequential(
                nn.Linear(3 * c3, dense), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(dense, emb_out), nn.ReLU(),
            )

        def forward(self, x):
            x = x.index_select(-1, self.perm)
            if self.is2d:
                x = x.reshape(x.shape[0], self.S, self.W, self.C)
            else:
                x = x.reshape(x.shape[0], self.S, self.W * self.C)
            f = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=-1)
            return self.head(f)

    class FvCNNEmbedding(nn.Module):
        """sbi's built-in ``CNNEmbedding`` wired for the fv grid (stats as channels).

        Gathers the flat fv into canonical stat-major order via ``perm`` and delegates
        to ``CNNEmbedding``, which auto-selects 1D vs 2D conv from ``input_shape``
        length and mixes the ``in_channels=S`` stat channels: 2D ``(W, C)`` for
        multi-scale, 1D ``(W*C,)`` for single-scale. A single-branch generic conv
        (no dilation ladder) — the structured-but-untuned control for the custom
        ``FvConvEmbedding`` tower; ``out_channels``/``kernel_size`` bump sbi's tiny
        defaults ([6, 12], k=5).
        """

        def __init__(self, S, W, C, perm, emb_out, out_channels, kernel_size):
            super().__init__()
            _, _, _, _, _, CNNEmbedding = _import_sbi()
            self.register_buffer("perm", torch.as_tensor(perm, dtype=torch.long))
            is2d = bool(W > 1 and C > 1)
            input_shape = (int(W), int(C)) if is2d else (int(W) * int(C),)
            self.cnn = CNNEmbedding(
                input_shape=input_shape,
                in_channels=int(S),
                out_channels_per_layer=list(out_channels),
                num_conv_layers=len(out_channels),
                output_dim=emb_out,
                kernel_size=kernel_size,
            )

        def forward(self, x):
            # CNNEmbedding reshapes the flat input to (batch, in_channels, *input_shape)
            # in channel-major raster order — exactly the order `perm` produces.
            return self.cnn(x.index_select(-1, self.perm))

    class FvCondEmbedding(nn.Module):
        """Wrap a base fv embedding and append known conditioning channels.

        The flow's input is ``x_aug = [fv | cond]`` where ``cond`` is the
        standardized log10 of known nuisance(s) (e.g. recombination rate ``r``).
        The first ``n_fv`` columns pass through the base embedding (fc/conv/cnn);
        the trailing ``n_cond`` columns **bypass** it and are concatenated to the
        learned summary, so the flow conditions on the EXACT known value —
        ``q(s,t | fv, r)`` — instead of marginalizing it. This mirrors ReLERNN's
        separate position branch (Min et al. 2026): the nuisance is side
        information, not a per-window statistic, so it must skip the conv tower.
        """

        def __init__(self, base, n_fv, n_cond):
            super().__init__()
            self.base = base
            self.n_fv = int(n_fv)
            self.n_cond = int(n_cond)

        def forward(self, x):
            e = self.base(x[..., :self.n_fv])
            cond = x[..., self.n_fv:self.n_fv + self.n_cond]
            return torch.cat([e, cond], dim=-1)

    class FvCausalCNNEmbedding(nn.Module):
        """sbi ``CausalCNNEmbedding`` on the fv grid — 1D dilated causal conv.

        Gathers the flat fv into stat-major order and treats it as a length-``W*C``
        sequence with ``in_channels=S`` stat channels; the causal conv's dilation
        ladder is the sbi-native analog of the custom ``FvConvEmbedding`` tower, so
        this is the fair control for "does the hand-built dilated conv earn its
        keep?". Like ``FvCNNEmbedding`` it passes the flat gathered vector and lets
        the sbi net reshape internally. ``pool_kernel_size`` is clamped to the
        sequence length so short single-scale grids do not under-run the pool.
        """

        def __init__(self, S, W, C, perm, emb_out, out_channels, kernel_size,
                     pool_kernel_size):
            super().__init__()
            from sbi.neural_nets.embedding_nets import CausalCNNEmbedding
            self.register_buffer("perm", torch.as_tensor(perm, dtype=torch.long))
            self.S, self.L = int(S), int(W) * int(C)
            pk = max(1, min(int(pool_kernel_size), self.L))
            self.net = CausalCNNEmbedding(
                input_shape=(self.L,), in_channels=self.S,
                out_channels_per_layer=list(out_channels),
                num_conv_layers=len(out_channels),
                kernel_size=kernel_size, pool_kernel_size=pk, output_dim=emb_out,
            )

        def forward(self, x):
            return self.net(x.index_select(-1, self.perm))

    class FvResNet2DEmbedding(nn.Module):
        """sbi ``ResNetEmbedding2D`` on the fv grid — deeper residual CNN.

        Reshapes the gathered fv to ``(batch, S, W, C)`` and runs a 2-D ResNet with
        ``c_in=S`` stat channels, ``c_out=emb_out``. Only meaningful on a genuine 2-D
        grid (``W>1`` and ``C>1``); the caller restricts it to multi-scale layouts and
        falls back to FC otherwise. Unlike the sbi CNN/CausalCNN nets this one needs an
        explicit 4-D input, so the reshape is done here rather than inside the net.
        """

        def __init__(self, S, W, C, perm, emb_out):
            super().__init__()
            from sbi.neural_nets.embedding_nets import ResNetEmbedding2D
            self.register_buffer("perm", torch.as_tensor(perm, dtype=torch.long))
            self.S, self.W, self.C = int(S), int(W), int(C)
            self.net = ResNetEmbedding2D(c_in=self.S, c_out=int(emb_out))

        def forward(self, x):
            x = x.index_select(-1, self.perm).reshape(x.shape[0], self.S, self.W, self.C)
            return self.net(x)

    class FvSpectralConvEmbedding(nn.Module):
        """sbi ``SpectralConvEmbedding`` (FNO-style) on the fv grid.

        Reshapes the gathered fv to ``(batch, S, W*C)`` and runs a spectral (Fourier)
        conv, well suited to the smooth diversity/LD spatial decay. The sbi net has no
        ``output_dim``; its flattened output length is measured once with a dummy
        forward and a ``Linear`` head maps it to ``emb_out``. ``modes`` is clamped to
        ``(W*C)//2`` so it never exceeds the available Fourier modes.
        """

        def __init__(self, S, W, C, perm, emb_out, modes, out_channels,
                     conv_channels, num_layers):
            super().__init__()
            from sbi.neural_nets.embedding_nets import SpectralConvEmbedding
            self.register_buffer("perm", torch.as_tensor(perm, dtype=torch.long))
            self.S, self.L = int(S), int(W) * int(C)
            m = max(1, min(int(modes), self.L // 2))
            self.net = SpectralConvEmbedding(
                in_channels=self.S, modes=m, out_channels=int(out_channels),
                conv_channels=int(conv_channels), num_layers=int(num_layers),
            )
            with torch.no_grad():
                d = self.net(torch.zeros(1, self.S, self.L)).reshape(1, -1).shape[-1]
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(d, int(emb_out)), nn.ReLU())

        def forward(self, x):
            x = x.index_select(-1, self.perm).reshape(x.shape[0], self.S, self.L)
            return self.head(self.net(x))

    # Make pickle resolve these as top-level module attributes (no <locals> qualname).
    for cls in (FvConvEmbedding, FvCNNEmbedding, FvCondEmbedding,
                FvCausalCNNEmbedding, FvResNet2DEmbedding, FvSpectralConvEmbedding):
        cls.__module__ = __name__
        cls.__qualname__ = cls.__name__
        _EMB_CLASSES[cls.__name__] = cls
    return _EMB_CLASSES


def __getattr__(name):  # PEP 562 — lazily expose the embedding classes for pickle
    if name in ("FvConvEmbedding", "FvCNNEmbedding", "FvCondEmbedding",
                "FvCausalCNNEmbedding", "FvResNet2DEmbedding",
                "FvSpectralConvEmbedding"):
        return _embedding_classes()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _make_conv_embedding(S, W, C, perm, *, emb_out, widths=(32, 64, 128),
                         dense=128, dropout=0.15):
    """Instantiate the custom three-branch dilated-conv embedding tower."""
    cls = _embedding_classes()["FvConvEmbedding"]
    return cls(S, W, C, perm, emb_out, tuple(widths), dense, dropout)


def _make_cnn_embedding(S, W, C, perm, *, emb_out,
                        out_channels=(32, 64), kernel_size=3):
    """Instantiate the sbi-``CNNEmbedding`` wrapper for the fv grid."""
    cls = _embedding_classes()["FvCNNEmbedding"]
    return cls(S, W, C, perm, emb_out, tuple(out_channels), kernel_size)


def _make_causalcnn_embedding(S, W, C, perm, *, emb_out,
                              out_channels=(32, 64), kernel_size=2,
                              pool_kernel_size=8):
    """Instantiate the sbi ``CausalCNNEmbedding`` wrapper (1D dilated causal conv)."""
    cls = _embedding_classes()["FvCausalCNNEmbedding"]
    return cls(S, W, C, perm, emb_out, tuple(out_channels), kernel_size,
               pool_kernel_size)


def _make_resnet2d_embedding(S, W, C, perm, *, emb_out):
    """Instantiate the sbi ``ResNetEmbedding2D`` wrapper (2-D residual CNN)."""
    cls = _embedding_classes()["FvResNet2DEmbedding"]
    return cls(S, W, C, perm, emb_out)


def _make_spectralconv_embedding(S, W, C, perm, *, emb_out, modes=8,
                                 out_channels=8, conv_channels=16, num_layers=3):
    """Instantiate the sbi ``SpectralConvEmbedding`` wrapper (FNO-style)."""
    cls = _embedding_classes()["FvSpectralConvEmbedding"]
    return cls(S, W, C, perm, emb_out, modes, out_channels, conv_channels,
               num_layers)


def _make_cond_embedding(base, n_fv, n_cond):
    """Wrap ``base`` so the flow conditions on appended standardized channels."""
    cls = _embedding_classes()["FvCondEmbedding"]
    return cls(base, n_fv, n_cond)


def _build_cond(df, cond_cols, transform=None, eps=1e-12):
    """Standardized ``log10`` conditioning matrix from known nuisance meta columns.

    Returns ``(cond_std float32 [n, k], transform, raw [n, k])``. At fit time
    (``transform=None``) it computes per-column ``log10`` mean/std and the raw range
    (for an extrapolation check); at predict time it applies the stored transform so
    the conditioning is on the SAME scale the flow was trained with. Recombination /
    mutation rates are strictly positive, so ``log10`` is safe (clipped at ``eps``).
    """
    raw = np.column_stack([np.asarray(df[c].to_numpy(), dtype=np.float64)
                           for c in cond_cols])
    lg = np.log10(np.clip(raw, eps, None))
    if transform is None:
        mu = lg.mean(0)
        sd = lg.std(0)
        sd[sd == 0] = 1.0
        transform = dict(cols=list(cond_cols), mu=mu, sd=sd,
                         raw_min=raw.min(0), raw_max=raw.max(0))
    else:
        mu = np.asarray(transform["mu"], dtype=np.float64)
        sd = np.asarray(transform["sd"], dtype=np.float64)
    cond_std = ((lg - mu) / sd).astype(np.float32)
    return cond_std, transform, raw


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
def fit_npe(
    parquet: str,
    artifact: str,
    *,
    targets: tuple = DEFAULT_TARGETS,
    s_transform: str = "natural",
    s_log_base: str = "e",
    calib_frac: float = 0.15,
    holdout_frac: float = 0.10,
    max_holdout: int = 4000,
    n_post: int = 1000,
    emb_out: int = 64,
    emb_hidden: int = 128,
    emb_layers: int = 3,
    hidden: int = 64,
    transforms: int = 5,
    flow: str = "nsf",
    z_score_theta: str = "independent",
    batch: int = 512,
    max_epochs: int = 300,
    n_pca: int = 50,
    ood_fit_rows: int = 30000,
    ood_quantile: float = 0.99,
    embedding: str = "fc",
    conv_widths: tuple = (32, 64, 128),
    conv_dense: int = 128,
    cnn_out_channels: tuple = (32, 64),
    spectral_modes: int = 8,
    condition_on: tuple = (),
    conformal: str = "global",
    conformal_t_edges: tuple = (1500.0, 3000.0),
    conformal_min_stratum: int = 30,
    seed: int = 42,
    device: str = "auto",
    data_device: str = "cpu",
) -> dict:
    """Train the flow, compute the global conformal radii + OOD gate, save artifact.

    Parameters
    ----------
    parquet : str
        Reference table with feature columns + ``s``/``t``/``model`` meta columns.
    artifact : str
        Output path for the pickled artifact (flow + conformal q + OOD gate).
    targets : tuple of str
        Which parameters the flow infers, in order. Default ``("s", "t")``. Append
        ``"pos"`` — the relative sweep position in [0.1, 0.9] of the 1.2 Mb locus —
        to also infer position: ``targets=("s", "t", "pos")``. The parquet must then
        carry a ``pos`` column; it is **opt-in** (not searched for by default). ``s``
        is converted from the stored ``-log(s)`` per ``s_transform``; ``t`` and
        ``pos`` are natural-scale.
    s_transform : {"natural", "neglog"}
        Units the flow learns / predicts ``s`` in. ``"natural"`` (default) decodes the
        stored ``-log(s)`` to the linear selection coefficient ``s``; ``"neglog"``
        trains directly on the stored ``-log(s)`` values as they are (base-agnostic,
        no decode). Saved in the artifact and replayed by :func:`predict_npe`, so the
        point estimates, credible intervals, and any passthrough true-``s`` column are
        all reported in the SAME units the flow was trained in.
    s_log_base : {"e", "10", number}
        Log base of the stored ``-log(s)`` column, used only when
        ``s_transform="natural"`` to invert the transform (``exp(-stored)`` for
        ``"e"``, ``base**(-stored)`` otherwise). Default ``"e"`` (i.e. stored
        ``-ln(s)``). Ignored for ``s_transform="neglog"``.
    z_score_theta : {"independent", "structured", "transform_to_unconstrained", "none"}
        How ``sbi`` standardizes the TARGET vector before the flow (passed through to
        ``posterior_nn``). This is preprocessing of ``theta`` only — it does **not**
        make the inference marginal: the flow still models the full JOINT
        ``q(s, t[, pos] | x)`` including off-diagonal dependence. ``"independent"``
        (default) standardizes each target dimension separately; ``"structured"``
        whitens with the full target covariance; ``"transform_to_unconstrained"``
        maps a bounded prior box to an unconstrained space (logit-type) before
        standardizing — the principled way to handle the bounded ``s`` box and to
        curb flow leakage past the edges; ``"none"`` disables it. Together with
        ``s_transform`` this is the knob for the s-parametrization / transform-
        invariance check (a flat ``s`` that is genuinely identifiability-limited is
        invariant to all of these).
    conformal : {"global","normalized","pred","conservative"}
        Conditional-conformal mode (see module docstring). ``"global"`` keeps the
        legacy single radius; the others adapt the radius per locus to even out
        per-subclass coverage, all using only deploy-available quantities. Replayed
        automatically by :func:`predict_npe` from the saved artifact.
    conformal_t_edges : tuple
        Predicted-``t`` bin edges (young/mid/old) for ``conformal="pred"``.
    conformal_min_stratum : int
        Minimum calib loci for a predicted stratum to get its own radius under
        ``conformal="pred"`` (else it falls back to the global radius).
    data_device : str
        Where the full ``(theta, x)`` training tensors are stored. ``"cpu"``
        (default) keeps them in host RAM and streams ``batch``-sized slices to the
        GPU each step, so VRAM scales with ``batch`` not row count — required for
        large reference tables (e.g. 200k rows x 5408 feats would otherwise pin
        ~4 GB in VRAM and OOM a 10 GB card). Set to ``"cuda"`` / the training device
        only for small tables where keeping data on-GPU avoids per-batch transfers.

    Returns
    -------
    dict
        The artifact dictionary that was saved to ``artifact``.
    """
    torch, NPE, posterior_nn, BoxUniform, FCEmbedding, CNNEmbedding = _import_sbi()
    dev = _resolve_device(device)
    torch.manual_seed(seed)
    if conformal not in CONFORMAL_MODES:
        raise ValueError(f"conformal={conformal!r} not in {CONFORMAL_MODES}")
    if s_transform not in S_TRANSFORMS:
        raise ValueError(f"s_transform={s_transform!r} not in {S_TRANSFORMS}")
    if s_transform == "natural":
        _s_base(s_log_base)  # validate the base is parseable
    targets = tuple(targets)
    s_units = ("-log(s) [as stored]" if s_transform == "neglog"
               else f"natural s (base={s_log_base})")
    logger.info("device=%s | theta=%s | s units: %s | t,pos NATURAL | conformal=%s",
                dev, targets, s_units, conformal)

    df = pl.read_parquet(parquet)
    miss_tgt = [c for c in targets if c not in df.columns]
    if miss_tgt:
        raise ValueError(
            f"targets={targets} but parquet is missing {miss_tgt}. "
            "For `pos`, add a 'pos' column (relative sweep position in [0.1, 0.9]).")
    feat = feature_cols(df)
    n_drop = len([c for c in df.columns if "delta_ihh" in c])
    logger.info("%d rows, %d features (dropped %d delta_ihh cols), targets=%s",
                df.height, len(feat), n_drop, targets)

    X = np.nan_to_num(df.select(feat).to_numpy().astype(np.float32),
                      nan=0.0, posinf=0.0, neginf=0.0)
    # per-target truth in training units (s per s_transform; t, pos pass through)
    truth = {nm: _decode_target(nm, df[nm].to_numpy(),
                                s_transform=s_transform, s_log_base=s_log_base)
             for nm in targets}
    model = df["model"].to_numpy().astype(str)
    theta = np.column_stack([truth[nm] for nm in targets]).astype(np.float32)

    # ---- optional conditioning on known nuisances (r, mu): q(s,t | x, r) ----
    # r is NOT a parameter (prior stays 2-D over s,t); it joins the flow's context.
    cond_cols = [c for c in condition_on]
    missing_cond = [c for c in cond_cols if c not in df.columns]
    if missing_cond:
        raise ValueError(f"condition_on columns missing from parquet: {missing_cond}")
    cond_transform = None
    if cond_cols:
        cond_std, cond_transform, _ = _build_cond(df, cond_cols)
        Xpost = np.concatenate([X, cond_std], axis=1).astype(np.float32)
        logger.info("conditioning flow on %s (log10 z-scored) -> x_aug dim %d "
                    "(%d fv + %d cond)", cond_cols, Xpost.shape[1], X.shape[1],
                    len(cond_cols))
    else:
        Xpost = X

    # train / calib (+ optional self-check holdout), all stratified by model
    if holdout_frac > 0:
        rest, hold = _stratified_split(model, holdout_frac, seed)
    else:
        rest = np.ones(len(model), bool)
        hold = np.zeros(len(model), bool)
    rest_idx = np.where(rest)[0]
    cal_within = _stratified_split(model[rest_idx], calib_frac, seed + 1)[1]
    cal = rest_idx[cal_within]
    tr = rest_idx[~cal_within]
    logger.info("train=%d calib=%d holdout=%d", len(tr), len(cal), int(hold.sum()))

    low = torch.from_numpy(theta.min(0) - 0.05 * np.abs(theta.min(0))).float()
    high = torch.from_numpy(theta.max(0) + 0.05 * np.abs(theta.max(0))).float()
    prior = BoxUniform(low=low, high=high, device=dev)
    emb_kind, grid = "fc", None
    if embedding in ("conv", "cnn", "causalcnn", "resnet2d", "spectralconv"):
        stats, wins, cens, perm = _parse_grid(feat)
        if perm is None:
            logger.warning(
                "embedding=%r requested but features are not a complete "
                "{stat}_{window}_{center} grid — falling back to FC embedding.",
                embedding)
        else:
            S, W, C = len(stats), len(wins), len(cens)
            grid = dict(S=S, W=W, C=C)
            is2d = bool(W > 1 and C > 1)
            if embedding == "conv":
                emb = _make_conv_embedding(S, W, C, perm, emb_out=emb_out,
                                           widths=tuple(conv_widths), dense=conv_dense)
                emb_kind = "conv2d" if is2d else "conv1d"
            elif embedding == "cnn":  # sbi built-in CNNEmbedding (structured control)
                emb = _make_cnn_embedding(S, W, C, perm, emb_out=emb_out,
                                          out_channels=tuple(cnn_out_channels))
                emb_kind = "sbicnn2d" if is2d else "sbicnn1d"
            elif embedding in ("causalcnn", "spectralconv", "resnet2d"):
                # sbi-native nets: some assert a minimum input size, so build inside
                # a guard and fall back to FC on a too-small / non-2-D layout, mirroring
                # the "not a complete grid -> FC" contract above.
                if embedding == "resnet2d" and not is2d:
                    logger.warning(
                        "embedding='resnet2d' needs a 2-D grid (W>1 and C>1); this "
                        "single-scale layout is 1-D — falling back to FC embedding.")
                else:
                    try:
                        if embedding == "causalcnn":  # native dilated conv (`conv` peer)
                            emb = _make_causalcnn_embedding(
                                S, W, C, perm, emb_out=emb_out,
                                out_channels=tuple(cnn_out_channels))
                        elif embedding == "spectralconv":  # FNO-style spectral conv
                            emb = _make_spectralconv_embedding(
                                S, W, C, perm, emb_out=emb_out, modes=spectral_modes)
                        else:  # resnet2d — 2-D residual CNN
                            emb = _make_resnet2d_embedding(S, W, C, perm, emb_out=emb_out)
                        emb_kind = embedding
                    except (AssertionError, ValueError, RuntimeError) as exc:
                        logger.warning(
                            "embedding=%r could not be built on this %dx%d grid (%s) "
                            "— falling back to FC embedding.", embedding, W, C, exc)
            if emb_kind != "fc":
                logger.info("%s embedding: S=%d W=%d C=%d -> %s (emb_out=%d)",
                            embedding, S, W, C, emb_kind, emb_out)
    if emb_kind == "fc":
        emb = FCEmbedding(input_dim=X.shape[1], output_dim=emb_out,
                          num_layers=emb_layers, num_hiddens=emb_hidden)
    if cond_cols:  # wrap base embedding so the trailing cond channels bypass it
        emb = _make_cond_embedding(emb, X.shape[1], len(cond_cols))
        emb_kind = emb_kind + "+cond"
    de = posterior_nn(model=flow, embedding_net=emb, z_score_theta=z_score_theta,
                      hidden_features=hidden, num_transforms=transforms)
    inf = NPE(prior=prior, density_estimator=de, device=dev)
    # data_device="cpu" keeps the full (theta, x) master tensors in host RAM; sbi
    # streams only `batch`-sized slices to the GPU each step (npe_base:707-710).
    # Without this, sbi's append_simulations defaults data_device to the training
    # device and copies the ENTIRE x (~n_train * n_feat * 4 B) into VRAM at once —
    # ~4 GB for 200k x 5408, which OOMs a 10 GB card. VRAM now scales with `batch`,
    # not row count. Pass data_device=dev to keep data on-GPU (faster, small sets).
    inf.append_simulations(
        torch.from_numpy(theta[tr]),
        torch.from_numpy(Xpost[tr]),
        data_device=data_device,
    )
    logger.info("training NPE ...")
    inf.train(training_batch_size=batch, max_num_epochs=max_epochs,
              stop_after_epochs=20, show_train_summary=True)
    post = inf.build_posterior()

    # ---- global conformal q on calib ----
    logger.info("sampling calib for conformal ...")
    dcal = _sample_batch(post, Xpost[cal], n_post, dev)  # list of (n_post, D) natural draws
    yc = {nm: truth[nm][cal] for nm in targets}
    qglobal = {}
    for idx, nm in enumerate(targets):
        for a in ALPHAS:
            lo, hi = _per_locus_quantiles(dcal, idx, a)
            qglobal[(nm, a)] = _cqr_q(lo, hi, yc[nm], a)
    for a in ALPHAS:
        logger.info("alpha=%.2f  %s", a,
                    "  ".join(f"q_{nm}={qglobal[(nm, a)]:.4g}" for nm in targets))

    # ---- conditional conformal structure (replayed at predict) ----
    conf = _build_conformal(conformal, dcal, yc, targets,
                            t_edges=conformal_t_edges, min_stratum=conformal_min_stratum)
    if conformal == "pred":
        logger.info("conformal=pred: t_edges=%s s_cut=%.4g min_stratum=%d",
                    conf["t_edges"], conf["s_cut"], conf["min_stratum"])
    elif conformal in ("normalized", "conservative"):
        logger.info("conformal=%s: q_t(0.9)=%.1f q_s(0.9)=%.4g",
                    conformal, conf["q"]["t|0.9"], conf["q"]["s|0.9"])

    # ---- OOD gate: PCA-whitened Mahalanobis on the sim-sweep manifold ----
    from sklearn.decomposition import PCA

    npca = min(n_pca, X.shape[1])
    sub = np.random.default_rng(seed).choice(
        tr, size=min(len(tr), ood_fit_rows), replace=False)
    pca = PCA(n_components=npca, whiten=True, random_state=seed)
    ztr = pca.fit_transform(X[sub])
    d_tr = (ztr ** 2).sum(1)
    ood_thr = float(np.quantile(d_tr, ood_quantile))
    logger.info("OOD gate: n_pca=%d thr(q%.2f)=%.1f", npca, ood_quantile, ood_thr)

    # move flow to CPU for portable reload
    try:
        post.potential_fn.device = "cpu"
    except Exception:
        pass

    art = dict(
        sbi_posterior=post,
        feat_cols=feat,
        targets=list(targets),
        qglobal={f"{nm}|{a}": v for (nm, a), v in qglobal.items()},
        conformal=conf,
        alphas=list(ALPHAS),
        theta_parametrization=dict(
            s=s_transform, t="natural", pos="natural",
            s_stored="-log(s)", s_transform=s_transform, s_log_base=str(s_log_base)),
        s_transform=s_transform, s_log_base=str(s_log_base),
        pca=pca, ood_thr=ood_thr, ood_quantile=ood_quantile,
        fv_mean=X.mean(0).astype(np.float64),
        fv_std=X.std(0).astype(np.float64),
        cond_cols=cond_cols, cond_transform=cond_transform,
        meta=dict(parquet=parquet, n_train=int(len(tr)), n_calib=int(len(cal)),
                  flow=flow, hidden=hidden, transforms=transforms,
                  z_score_theta=z_score_theta,
                  embedding=emb_kind, grid=grid, condition_on=cond_cols,
                  conformal=conformal),
    )
    save_pickle(artifact, art)
    logger.info("artifact -> %s", artifact)

    # ---- optional marginal self-check on the holdout (post-conformal) ----
    if holdout_frac > 0 and hold.sum() > 0:
        hi_idx = np.where(hold)[0]
        if len(hi_idx) > max_holdout:
            hi_idx = np.random.default_rng(seed).choice(hi_idx, max_holdout, replace=False)
        logger.info("self-check: sampling %d holdout loci ...", len(hi_idx))
        dh = _sample_batch(post, Xpost[hi_idx], n_post, dev)  # list of (n_post, D)
        tidx = {nm: i for i, nm in enumerate(targets)}
        s_med_h = np.array([np.median(d[:, tidx["s"]]) for d in dh]) if "s" in tidx else None
        t_med_h = np.array([np.median(d[:, tidx["t"]]) for d in dh]) if "t" in tidx else None
        sig_h = {nm: np.array([d[:, tidx[nm]].std() + 1e-12 for d in dh]) for nm in targets}
        qg90 = {f"{nm}|0.9": qglobal[(nm, 0.90)] for nm in targets}
        check = {}
        for nm in targets:
            lo, hi = _per_locus_quantiles(dh, tidx[nm], 0.90)
            rad = _conformal_radius(conf, qg90, nm, 0.90, len(hi_idx),
                                    s_med=s_med_h, t_med=t_med_h, sigma=sig_h[nm])
            loc, hic = lo - rad, hi + rad
            y = truth[nm][hi_idx]
            cov = float(np.mean((y >= loc) & (y <= hic)))
            w = float(np.mean(hic - loc))
            check[nm] = dict(cov90=cov, width90=w)
            logger.info("self-check %s [%s]: cov90=%.3f width90=%.4g",
                        nm, conformal, cov, w)
        art["self_check"] = check

    return art


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
def predict_npe(
    artifact: str,
    candidates: str,
    out: str | None = None,
    *,
    n_post: int = 1000,
    max_rows: int = 0,
    norm_warn: float = 0.5,
    device: str = "auto",
) -> "pl.DataFrame":
    """Score candidate windows with a saved artifact.

    Returns a DataFrame with point estimates, conformal marginal CIs at each
    ALPHA, and per-locus OOD flags; passthrough non-feature columns are kept.
    """
    _import_sbi()  # ensure torch is importable for sampling
    dev = _resolve_device(device)
    art = load_pickle(artifact)
    feat = art["feat_cols"]
    post = art["sbi_posterior"]
    targets = tuple(art.get("targets", DEFAULT_TARGETS))  # older artifacts: (s, t)
    # s units the flow was trained in (older artifacts predate this -> natural, -ln)
    s_transform = art.get("s_transform", "natural")
    s_log_base = art.get("s_log_base", "e")
    qglobal = art["qglobal"]  # {"t|0.9": q, ...} — global radii / legacy fallback
    conf = art.get("conformal") or {"mode": "global"}
    alphas = tuple(art["alphas"])
    logger.info("targets=%s | s units: %s | conformal mode: %s", targets,
                ("-log(s)" if s_transform == "neglog"
                 else f"natural (base={s_log_base})"), conf.get("mode", "global"))
    cond_cols = art.get("cond_cols", []) or []
    cond_transform = art.get("cond_transform")
    logger.info("loaded artifact: %d features, alphas=%s, conditioning=%s, device=%s",
                len(feat), alphas, cond_cols or "none", dev)

    df = pl.read_parquet(candidates)
    missing = [c for c in feat if c not in df.columns]
    if missing:
        raise ValueError(
            f"candidate parquet missing {len(missing)} required feature columns, "
            f"e.g. {missing[:5]}"
        )
    id_cols = [c for c in df.columns if c not in feat]
    if max_rows and df.height > max_rows:
        df = df.head(max_rows)
        logger.info("capped to first %d rows", max_rows)

    X = np.nan_to_num(df.select(feat).to_numpy().astype(np.float32),
                      nan=0.0, posinf=0.0, neginf=0.0)
    logger.info("%d candidate windows, %d features", X.shape[0], X.shape[1])

    # conditioning channels (r, mu): must be present + on the trained scale
    cond_oob = np.zeros(X.shape[0], bool)
    if cond_cols:
        miss_c = [c for c in cond_cols if c not in df.columns]
        if miss_c:
            raise ValueError(
                f"artifact conditions on {cond_cols} but candidate parquet is "
                f"missing {miss_c} (e.g. the per-region recombination rate).")
        cond_std, _, cond_raw = _build_cond(df, cond_cols, transform=cond_transform)
        Xpost = np.concatenate([X, cond_std], axis=1).astype(np.float32)
        lo_r = np.asarray(cond_transform["raw_min"], dtype=np.float64)
        hi_r = np.asarray(cond_transform["raw_max"], dtype=np.float64)
        cond_oob = ((cond_raw < lo_r) | (cond_raw > hi_r)).any(axis=1)
        if cond_oob.any():
            logger.warning(
                "%d/%d candidates have %s outside the training range (the flow "
                "extrapolates there — widen the simulation r-prior to cover the map).",
                int(cond_oob.sum()), len(cond_oob), cond_cols)
        logger.info("conditioning on %s -> x_aug dim %d", cond_cols, Xpost.shape[1])
    else:
        Xpost = X

    # normalization-reference guard
    mshift = float(np.mean(np.abs(X.mean(0) - art["fv_mean"])))
    sratio = float(np.median((art["fv_std"] + 1e-9) / (X.std(0) + 1e-9)))
    logger.info("norm-guard: mean|dmu|=%.3f median sd ratio(sim/cand)=%.2f",
                mshift, sratio)
    if mshift > norm_warn:
        logger.warning(
            "candidate fv mean differs markedly from the training reference — "
            "verify the SAME normalization reference was used (mean|dmu|=%.3f).",
            mshift,
        )

    # OOD per candidate
    pca = art["pca"]
    d_e = (pca.transform(X) ** 2).sum(1)
    ood = d_e > art["ood_thr"]
    logger.info("OOD: %d/%d (%.1f%%) beyond sim q%.2f",
                int(ood.sum()), len(ood), 100 * ood.mean(), art["ood_quantile"])

    # sample posteriors (x_aug carries the conditioning channels when enabled)
    logger.info("sampling posteriors ...")
    draws = _sample_batch(post, Xpost, n_post, dev)  # list of (n_post, D) natural draws
    tidx = {nm: i for i, nm in enumerate(targets)}
    comp = {nm: [d[:, i] for d in draws] for nm, i in tidx.items()}  # per-target draw lists

    # passthrough id columns; convert a true-s column (stored -log(s)) into the SAME
    # units the flow was trained in, so it matches s_point (labeled tables only; real
    # candidates have no s column). true `t`/`pos` pass through unchanged.
    cols = {c: (_decode_s(df[c].to_numpy(), transform=s_transform, base=s_log_base)
                if c == "s" else df[c]) for c in id_cols}
    med = {nm: np.array([float(np.median(v)) for v in comp[nm]]) for nm in targets}
    sigma = {nm: np.array([v.std() + 1e-12 for v in comp[nm]]) for nm in targets}
    for nm in targets:
        cols[f"{nm}_point"] = med[nm]
    s_med = med.get("s")   # strata keys for pred/normalized conformal (if present)
    t_med = med.get("t")
    n = X.shape[0]
    for nm in targets:
        P = comp[nm]
        for a in alphas:
            ql, qh = (1 - a) / 2, (1 + a) / 2
            rad = _conformal_radius(conf, qglobal, nm, a, n,
                                    s_med=s_med, t_med=t_med, sigma=sigma[nm])
            lo = np.array([np.quantile(P[i], ql) for i in range(n)]) - rad
            hi = np.array([np.quantile(P[i], qh) for i in range(n)]) + rad
            A = int(a * 100)
            cols[f"{nm}_lo{A}"] = lo
            cols[f"{nm}_hi{A}"] = hi
    cols["ood_flag"] = ood
    cols["ood_dist"] = d_e
    if cond_cols:
        cols["cond_oob"] = cond_oob

    res = pl.DataFrame(cols)
    if out:
        res.write_parquet(out)
        logger.info("predictions -> %s (%d rows, %d cols)",
                    out, res.height, len(res.columns))
    return res


# ---------------------------------------------------------------------------
# sbi-native calibration + analysis (SBC / TARP / rank + pair plots)
# ---------------------------------------------------------------------------
def analyze_npe(
    artifact: str,
    parquet: str,
    outdir: str | None = None,
    *,
    n_rows: int = 2000,
    n_post: int = 1000,
    pairplot_row: int | None = 0,
    seed: int = 42,
    device: str = "auto",
) -> dict:
    """Run sbi's own default calibration/analysis on a saved NPE artifact.

    This is the calibration-DIAGNOSTIC entry point (distinct from the conformal
    CORRECTION applied in :func:`predict_npe`). On a LABELED table (true target
    columns present) it exercises sbi's built-in tools directly:

      * **SBC** (Talts et al. 2018) — ``run_sbc`` + ``check_sbc``: posterior-rank
        uniformity per target, with KS p-values, C2ST(ranks) and C2ST of the
        data-averaged posterior. Non-uniform ranks reveal bias/over- or
        under-dispersion of the RAW flow (before any conformal widening).
      * **TARP** (Lemos et al. 2023) — ``run_tarp`` + ``check_tarp``: expected
        coverage vs credibility, area-to-curve (``atc``, 0 = ideal) and KS p-value.
        Skipped with a warning if the installed sbi predates TARP.

    When ``outdir`` is given, saves sbi's default ``sbc_rank_plot`` (rank
    histograms), a TARP expected-coverage curve, and a single-locus joint
    ``pairplot`` (row ``pairplot_row``, truth overlaid) — the natural figure for
    the joint ``(s, t[, pos])`` posterior.

    Parameters
    ----------
    artifact : str
        Path to a :func:`fit_npe` artifact (carries the sbi posterior + targets).
    parquet : str
        LABELED table with the feature columns AND the true target columns. Use a
        held-out reference split, never the raw candidate windows.
    n_rows : int
        Subsample this many loci for the diagnostics (SBC/TARP resample the
        posterior per locus, so this bounds runtime). ``0`` uses all rows.
    n_post : int
        Posterior draws per locus.
    pairplot_row : int or None
        Row index to draw the example joint posterior for; ``None`` skips it.
    device : {"auto", "cpu", "cuda"}
        ``"auto"`` (default) runs the diagnostics on whatever device the loaded
        flow already lives on (avoids a CPU/CUDA mismatch when scoring a
        GPU-trained artifact); an explicit value moves the flow there first.

    Returns
    -------
    dict
        ``{"targets", "n_rows", "n_post", "sbc": {...}, "tarp": {...}?}`` — the
        JSON-able diagnostic summary (also logged).
    """
    torch = _import_sbi()[0]
    (run_sbc, check_sbc, run_tarp, check_tarp,
     sbc_rank_plot, pairplot) = _import_sbi_analysis()

    art = load_pickle(artifact)
    post = art["sbi_posterior"]
    # Align to the flow's device. Default "auto" = wherever the (possibly
    # GPU-trained) flow already sits; an explicit device moves the flow there.
    if device == "auto":
        dev = _posterior_device(post)
    else:
        dev = _resolve_device(device)
        try:
            post.to(dev)
        except Exception as exc:  # pragma: no cover
            logger.warning("posterior.to(%s) failed (%s); using its current device", dev, exc)
            dev = _posterior_device(post)
    logger.info("running sbi diagnostics on device=%s", dev)
    feat = art["feat_cols"]
    targets = tuple(art.get("targets", DEFAULT_TARGETS))
    s_transform = art.get("s_transform", "natural")
    s_log_base = art.get("s_log_base", "e")
    cond_cols = art.get("cond_cols", []) or []
    cond_transform = art.get("cond_transform")

    df = pl.read_parquet(parquet)
    miss = [c for c in feat if c not in df.columns]
    if miss:
        raise ValueError(
            f"parquet missing {len(miss)} feature columns, e.g. {miss[:5]}")
    miss_t = [nm for nm in targets if nm not in df.columns]
    if miss_t:
        raise ValueError(
            f"analyze_npe needs the TRUE target columns {miss_t} — run it on a "
            "labeled reference/holdout table, not on raw candidate windows.")
    if n_rows and df.height > n_rows:
        df = df.sample(n=n_rows, seed=seed)
        logger.info("subsampled %d loci for SBC/TARP", n_rows)

    X = np.nan_to_num(df.select(feat).to_numpy().astype(np.float32),
                      nan=0.0, posinf=0.0, neginf=0.0)
    truth = {nm: _decode_target(nm, df[nm].to_numpy(),
                                s_transform=s_transform, s_log_base=s_log_base)
             for nm in targets}
    theta = np.column_stack([truth[nm] for nm in targets]).astype(np.float32)
    if cond_cols:  # same conditioning channels the flow was trained with
        cond_std, _, _ = _build_cond(df, cond_cols, transform=cond_transform)
        X = np.concatenate([X, cond_std], axis=1).astype(np.float32)

    thetas = torch.from_numpy(theta).to(dev)
    xs = torch.from_numpy(X).to(dev)
    logger.info("sbi calibration: %d loci, %d post draws, targets=%s",
                len(thetas), n_post, targets)

    # ---- SBC (Talts 2018): rank uniformity of the RAW flow posterior ----
    ranks, dap = run_sbc(thetas, xs, post, num_posterior_samples=n_post)
    ranks, dap = ranks.cpu(), dap.cpu()  # check_sbc / plotting stay on CPU
    sbc_raw = check_sbc(ranks, thetas.cpu(), dap, num_posterior_samples=n_post)
    sbc = {k: _to_list(v) for k, v in dict(sbc_raw).items()}
    for k, v in sbc.items():
        logger.info("SBC %-14s %s", k, np.round(np.asarray(v, float), 4))
    out = dict(targets=list(targets), n_rows=int(len(thetas)),
               n_post=int(n_post), sbc=sbc)

    # ---- TARP expected coverage (Lemos 2023), if the installed sbi has it ----
    ecp = alpha = None
    if run_tarp is not None:
        try:
            ecp, alpha = run_tarp(thetas, xs, post, num_posterior_samples=n_post)
            ecp, alpha = ecp.cpu(), alpha.cpu()
            tarp = dict(ecp=_to_list(ecp), alpha=_to_list(alpha))
            if check_tarp is not None:
                atc, ks_p = check_tarp(ecp, alpha)
                tarp.update(atc=float(atc), ks_pval=float(ks_p))
                logger.info("TARP ATC=%.4f (0=ideal)  ks_pval=%.3f",
                            float(atc), float(ks_p))
            out["tarp"] = tarp
        except Exception as exc:  # pragma: no cover
            logger.warning("TARP diagnostic failed (%s); skipping.", exc)
    else:
        logger.info("installed sbi has no TARP — skipping expected-coverage curve.")

    # ---- sbi's default plots ----
    if outdir:
        import os

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(outdir, exist_ok=True)
        if sbc_rank_plot is not None:
            try:
                fig, _ = sbc_rank_plot(ranks=ranks, num_posterior_samples=n_post,
                                       plot_type="hist",
                                       parameter_labels=list(targets))
                p = os.path.join(outdir, "sbc_rank_hist.png")
                fig.savefig(p, dpi=150); plt.close(fig)
                logger.info("SBC rank plot -> %s", p)
            except Exception as exc:  # pragma: no cover
                logger.warning("sbc_rank_plot failed (%s)", exc)
        if ecp is not None and alpha is not None:
            fig, a = plt.subplots(figsize=(5.2, 5.0))
            a.plot(_to_list(alpha), _to_list(ecp), "o-", c="#3a7ca5")
            a.plot([0, 1], [0, 1], "--", c="grey", lw=1)
            a.set_xlim(0, 1); a.set_ylim(0, 1)
            a.set_xlabel("credibility level"); a.set_ylabel("expected coverage")
            a.set_title("TARP expected coverage")
            p = os.path.join(outdir, "tarp_coverage.png")
            fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
            logger.info("TARP coverage plot -> %s", p)
        if (pairplot is not None and pairplot_row is not None
                and 0 <= pairplot_row < len(xs)):
            try:
                smp = post.sample((n_post,), x=xs[pairplot_row],
                                  show_progress_bars=False).cpu().numpy()
                fig, _ = pairplot(smp, labels=list(targets),
                                  points=theta[pairplot_row][None, :])
                p = os.path.join(outdir, f"pairplot_row{pairplot_row}.png")
                fig.savefig(p, dpi=150); plt.close(fig)
                logger.info("joint pairplot -> %s", p)
            except Exception as exc:  # pragma: no cover
                logger.warning("pairplot failed (%s)", exc)

    return out


# ---------------------------------------------------------------------------
# Evaluation plots (publication Figs 1-3)
# ---------------------------------------------------------------------------
def plot_evaluation(
    predictions,
    outdir: str,
    *,
    subclass_col: str = "model",
    s_label: str = "s (selection coefficient)",
    abcrf_t_width90: float = 2372.0,  # ABC-RF·profile t width-90 on the unified 800-test
    bin_by: tuple[str, ...] = ("t", "s"),
    bin_edges: dict | None = None,
    bin_min_n: int = 50,
    bin_targets: dict | None = None,
    bin_ceil: tuple[str, ...] = ("t",),
) -> None:
    """Publication evaluation figures from a labeled :func:`predict_npe` output.

    ``predictions`` is a DataFrame (or parquet path) that retains the TRUE ``s``,
    ``t`` and subclass columns — i.e. run ``predict_npe`` on the per-subclass
    labeled table, whose meta columns pass through.

    Each scalar metric is its own PNG (one sub-panel per target), plotted as a bar
    chart whose x-axis walks the four grouping levels in ordered, color-coded
    sections, so you read how the metric behaves across every split at a glance:

        global | young · old | hard · soft | <8 subclasses>

    Files written to ``outdir``:
      fig_mae.png             mean absolute error (recovery)
      fig_rmse.png            root-mean-square error (recovery)
      fig_coverage.png        empirical 90% coverage per group (±binomial SE, 0.90 line)
      fig_sharpness.png       mean 90% interval width (``t`` panel: ABC-RF line)
      fig_calibration.png     empirical vs nominal coverage curve (pooled, per target)
      fig_recovery_scatter.png  global predicted-vs-true scatter (one panel per target,
                                colored by target; MAE/RMSE/r in the title)
      fig_recovery_{split}_{target}.png  split recovery, a SEPARATE file per target
                                (t / s) so the two never share a panel; one panel per
                                group on native axes with MAE/RMSE/r in the title.
                                split in {subclass, hardsoft, youngold}
      fig_vs_{var}_{metric}.png  BINNED performance, ONE metric per file (metric in
                                {mae, rmse, width, coverage}) for each ``var`` in
                                ``bin_by``. The metric as a line over LABELLED,
                                equally-spaced range categories ("0–100", "100–250", …)
                                with an aligned n-per-bin histogram strip beneath, so
                                you read "from X to Y it works this well" directly.
                                Each var is paired with its own target, so the file is
                                ``fig_vs_t_*`` (t target) and ``fig_vs_s_*`` (s target);
                                a non-target var (f_i, f_t) or an explicit cross-pair
                                names the target too: ``fig_vs_f_t_t_coverage.png``.

    The young/old and hard/soft sections are derived by substring match on
    ``subclass_col`` (rows lacking the token — e.g. a neutral class — are omitted
    from that section only); absent sections are skipped. With no ``subclass_col``
    only the global bar is drawn. Figs for joint HPD, identifiability and domain/OOD
    need extra inputs and live in the option scripts (see report §8).

    Binned continuous report (``fig_vs_{var}.png``)
    -----------------------------------------------
    ``bin_by`` lists the TRUE meta columns to bin the x-axis on (default
    ``("t", "s")``; add ``"f_t"``, ``"f_i"``, ``"pos"`` as needed). Each variable
    gets ordered, interpretable edges — from ``bin_edges[var]`` if given, else a
    sensible per-variable default clipped to the data range — and any run of fine
    bins whose pooled count is below ``bin_min_n`` is greedily merged with its
    neighbour, so no plotted point rests on too few loci (a sparse old-sweep tail
    collapses to one wide bin rather than a noisy scatter). Binning is on the true
    value (an evaluation grouping, exactly like the subclass sections); the point
    x-position is the in-bin median so it sits where the data actually is.

    One metric per file (four files per var·target: mae, rmse, width, coverage), so
    figures slot into a manuscript individually. Each var is paired with its own
    target — ``bin_targets`` overrides this ``{var: [targets]}`` (e.g. show ``t`` and
    ``s`` binned along ``f_t``: ``bin_targets={"f_t": ["t", "s"]}``); a non-target var
    defaults to the ``t`` target. The x-axis is an ORDINAL category axis: one
    equally-spaced slot per bin, labelled with the bin's range only — the same
    labelled-category style as the subclass bar figures — and every metric figure
    carries an aligned n-per-bin histogram strip beneath (same bins, same order) so
    sample size is legible everywhere. ``bin_ceil`` lists vars to ``np.ceil`` before
    binning (default ``("t",)``: sim generation-times are floats, so ceiling yields
    integer ages and clean round edges). Read a threshold off the coverage line ("holds
    ≥0.90 up to the 1000–1500 bin") and watch width/MAE grow across the categories.
    """
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pl.read_parquet(predictions) if isinstance(predictions, str) else predictions
    os.makedirs(outdir, exist_ok=True)
    # targets present in the table = any column with both a true column and a *_point
    # column (derived from the data, not a fixed tuple, so any fit_npe target — s, t,
    # pos, f_t, f_i, … — is plotted). ``labels``/``palette`` carry nice captions for the
    # known ones and fall back to the bare name via ``.get`` for anything else.
    labels = {"t": "t (generations)", "s": s_label, "pos": "pos (relative position)",
              "f_t": "f_t (final frequency)", "f_i": "f_i (initial frequency)"}
    tgs = [c[:-len("_point")] for c in df.columns
           if c.endswith("_point") and c[:-len("_point")] in df.columns]
    if not tgs:
        raise ValueError("predictions missing true/point columns — run predict_npe on a "
                         "labeled table that keeps the true target columns")
    alphas = sorted(int(c[len(tgs[0]) + 3:]) for c in df.columns
                    if c.startswith(f"{tgs[0]}_lo") and c[len(tgs[0]) + 3:].isdigit())
    A90 = 90 if 90 in alphas else (max(alphas) if alphas else 90)
    palette = {"t": "#3a7ca5", "s": "#e9c46a", "pos": "#8ab17d",
               "f_t": "#8a5a83", "f_i": "#c1666b"}

    # cache truth / point / interval arrays once (float ndarrays, indexed by mask)
    tru = {nm: df[nm].to_numpy().astype(float) for nm in tgs}
    pnt = {nm: df[f"{nm}_point"].to_numpy().astype(float) for nm in tgs}
    lo = {(nm, A): df[f"{nm}_lo{A}"].to_numpy().astype(float)
          for nm in tgs for A in alphas}
    hi = {(nm, A): df[f"{nm}_hi{A}"].to_numpy().astype(float)
          for nm in tgs for A in alphas}

    def _cov(mask, nm, A):
        y, l, h = tru[nm][mask], lo[(nm, A)][mask], hi[(nm, A)][mask]
        return float(np.mean((y >= l) & (y <= h))) if y.size else float("nan")

    def _width(mask, nm, A):
        w = (hi[(nm, A)] - lo[(nm, A)])[mask]
        return float(np.mean(w)) if w.size else float("nan")

    def _group_colors(gs):
        cmap = plt.get_cmap("tab10" if len(gs) <= 10 else "tab20")
        return {g: cmap(i % cmap.N) for i, g in enumerate(gs)}

    # ------------------------------------------------------------------
    # Assemble the x-axis: one bar per group, laid out in ordered SECTIONS
    # (global | young/old | hard/soft | subclass). Each bar carries a boolean
    # mask into the full cached arrays, so every metric is just a reduction of
    # that mask. Sections that are absent (no subclass column, missing token) are
    # simply skipped.
    # ------------------------------------------------------------------
    n = df.height
    sec_color = {"global": "#6c757d", "age": "#3a7ca5",
                 "mode": "#2a9d8f", "subclass": "#e9c46a"}
    sections: list[tuple[str, list[tuple[str, np.ndarray]]]] = [
        ("global", [("global", np.ones(n, bool))])]

    if subclass_col in df.columns:
        model = df[subclass_col].to_numpy().astype(str)

        def _token(match_to):
            """Full-length substring label (None where no token matches)."""
            return np.array([next((t for t in match_to if t in m), None)
                             for m in model], dtype=object)

        age = _token(("young", "old"))
        age_bars = [(g, age == g) for g in ("young", "old") if (age == g).any()]
        if age_bars:
            sections.append(("age", age_bars))
        mode = _token(("hard", "soft"))
        mode_bars = [(g, mode == g) for g in ("hard", "soft") if (mode == g).any()]
        if mode_bars:
            sections.append(("mode", mode_bars))
        subclasses = sorted(np.unique(model).tolist())
        sections.append(("subclass", [(g, model == g) for g in subclasses]))
    else:
        model = None
        logger.warning("no %r column — plotting the global bar only", subclass_col)

    # flatten to parallel arrays + record section spans for separators/legend
    bar_labels, bar_masks, bar_colors, spans = [], [], [], []
    for sec, blist in sections:
        start = len(bar_labels)
        for lab, m in blist:
            bar_labels.append(lab); bar_masks.append(m)
            bar_colors.append(sec_color.get(sec, "#888888"))
        spans.append((sec, start, len(bar_labels)))
    x = np.arange(len(bar_labels))

    def _metric(mask, nm, which):
        if which == "mae":
            e = pnt[nm][mask] - tru[nm][mask]
            return float(np.mean(np.abs(e))) if e.size else float("nan")
        if which == "rmse":
            e = pnt[nm][mask] - tru[nm][mask]
            return float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan")
        if which == "coverage":
            return _cov(mask, nm, A90)
        return _width(mask, nm, A90)  # sharpness

    from matplotlib.patches import Patch
    sec_legend = [Patch(facecolor=sec_color[s], label=s) for s, _, _ in spans]

    def _metric_fig(which, ylabel, *, hline=None, ylim=None, ref_t=None):
        """One PNG for ``which`` metric: sub-panel per target, bars across sections."""
        fig, ax = plt.subplots(1, len(tgs), squeeze=False,
                               figsize=(max(6.0, 0.42 * len(bar_labels)) * len(tgs), 5.2))
        for j, nm in enumerate(tgs):
            a = ax[0][j]
            vals = [_metric(m, nm, which) for m in bar_masks]
            yerr = None
            if which == "coverage":  # binomial SE per group: sqrt(p(1-p)/n)
                yerr = [np.sqrt(max(v, 0) * (1 - max(v, 0)) / max(int(m.sum()), 1))
                        for v, m in zip(vals, bar_masks)]
            a.bar(x, vals, 0.8, color=bar_colors, alpha=0.88, yerr=yerr, capsize=2)
            for _, st, en in spans:  # dotted separators between sections
                if st > 0:
                    a.axvline(st - 0.5, color="grey", lw=0.7, ls=":")
            if hline is not None:
                a.axhline(hline, ls="--", c="#d1495b", lw=1)
            if ref_t is not None and nm == "t":
                a.axhline(ref_t, ls="--", c="k", lw=1, label=f"ABC-RF {ref_t:.0f}")
                a.legend(fontsize=7, loc="upper right")
            if ylim is not None:
                a.set_ylim(*ylim)
            a.set_xticks(x)
            a.set_xticklabels(bar_labels, rotation=90, fontsize=7)
            a.set_ylabel(f"{labels.get(nm, nm)} — {ylabel}")
            a.set_title(nm)
        ax[0][0].legend(handles=sec_legend, fontsize=7, loc="upper left", title="grouping")
        fig.suptitle(f"{ylabel} across groupings", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        path = os.path.join(outdir, f"fig_{which}.png")
        fig.savefig(path, dpi=150); plt.close(fig)
        logger.info("evaluation figure -> %s", path)

    _metric_fig("mae", "MAE")
    _metric_fig("rmse", "RMSE")
    # coverage floor adapts so a poorly-covered group is never clipped out of view
    cov_min = min((_metric(m, nm, "coverage") for m in bar_masks for nm in tgs),
                  default=0.6)
    cov_floor = min(0.6, np.floor(cov_min * 20) / 20) if np.isfinite(cov_min) else 0.6
    _metric_fig("coverage", f"{A90}% coverage (±binomial SE)",
                hline=A90 / 100, ylim=(cov_floor, 1.0))
    _metric_fig("sharpness", f"{A90}% interval width", ref_t=abcrf_t_width90)

    # ---- extra: calibration curve (empirical vs nominal coverage) for each target ----
    # pooled over all candidates; one line per target, diagonal = perfect calibration.
    fig, a = plt.subplots(figsize=(6.5, 5.2))
    xs = [al / 100 for al in alphas]
    for nm in tgs:
        ys = [_cov(slice(None), nm, al) for al in alphas]
        a.plot(xs, ys, "o-", c=palette.get(nm, "#3a7ca5"), label=nm)
    a.plot([0, 1], [0, 1], "--", c="grey", lw=1)
    a.set_xlim(0, 1); a.set_ylim(0, 1)
    a.set_xlabel("nominal coverage"); a.set_ylabel("empirical coverage")
    a.set_title("Calibration (pooled, marginal)"); a.legend(title="target")
    fig.tight_layout()
    path = os.path.join(outdir, "fig_calibration.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    logger.info("evaluation figure -> %s", path)

    def _mae_rmse_r(mask, nm):
        yt, yp = tru[nm][mask], pnt[nm][mask]
        if yt.size == 0:
            return float("nan"), float("nan"), float("nan")
        mae = float(np.mean(np.abs(yp - yt)))
        rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
        r = float(np.corrcoef(yt, yp)[0, 1]) if yt.size > 1 else float("nan")
        return mae, rmse, r

    # ---- global recovery scatter: one panel per target, colored by target ----
    allm = np.ones(n, bool)
    fig, ax = plt.subplots(1, len(tgs), figsize=(5.5 * len(tgs), 5), squeeze=False)
    for j, nm in enumerate(tgs):
        a = ax[0][j]
        a.scatter(tru[nm], pnt[nm], s=10, alpha=0.35, linewidths=0,
                  color=palette.get(nm, "#3a7ca5"))
        lim = [min(tru[nm].min(), pnt[nm].min()), max(tru[nm].max(), pnt[nm].max())]
        a.plot(lim, lim, "--", c="grey", lw=1)
        mae, rmse, r = _mae_rmse_r(allm, nm)
        a.set_xlabel(f"true {labels.get(nm, nm)}"); a.set_ylabel(f"predicted {labels.get(nm, nm)}")
        a.set_title(f"{nm}: MAE={mae:.3g}  RMSE={rmse:.3g}  r={r:.3f}")
    fig.suptitle("Recovery — predicted vs true (global)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(outdir, "fig_recovery_scatter.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    logger.info("evaluation figure -> %s", path)

    # ---- split recovery: SEPARATE file per (split, target); one panel per group,
    # native axes, MAE/RMSE/r in each panel title. ----
    def _recovery_split(fname, title, blist, nm):
        ng = len(blist)
        ncols = min(4, ng); nrows = int(np.ceil(ng / ncols))
        fig, ax = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 4.1 * nrows),
                               squeeze=False)
        for idx, (lab, m) in enumerate(blist):
            a = ax[idx // ncols][idx % ncols]
            yt, yp = tru[nm][m], pnt[nm][m]
            a.scatter(yt, yp, s=10, alpha=0.4, linewidths=0,
                      color=palette.get(nm, "#3a7ca5"))
            if yt.size:
                lim = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
                a.plot(lim, lim, "--", c="grey", lw=1)
            mae, rmse, r = _mae_rmse_r(m, nm)
            a.set_xlabel(f"true {labels.get(nm, nm)}"); a.set_ylabel(f"predicted {labels.get(nm, nm)}")
            a.set_title(f"{lab} (n={int(m.sum())})\nMAE={mae:.3g} RMSE={rmse:.3g} r={r:.2f}",
                        fontsize=9)
        for k in range(ng, nrows * ncols):  # hide unused axes
            ax[k // ncols][k % ncols].axis("off")
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        path = os.path.join(outdir, fname)
        fig.savefig(path, dpi=150); plt.close(fig)
        logger.info("evaluation figure -> %s", path)

    sec_dict = {sec: blist for sec, blist in sections}
    for split, sec in (("subclass", "subclass"), ("hardsoft", "mode"),
                       ("youngold", "age")):
        blist = sec_dict.get(sec)
        if not blist:
            continue
        for nm in tgs:  # one file per target so t and s never share a panel
            _recovery_split(f"fig_recovery_{split}_{nm}.png",
                            f"Recovery {nm} — {split}", blist, nm)

    # ------------------------------------------------------------------
    # Binned report: ONE metric per file, fig_vs_{var}[_{nm}]_{metric}.png.
    # Bin the TRUE value of each `var` in `bin_by` into round, interpretable range
    # categories (greedy adaptive merge so no bin < `bin_min_n`), then reduce the SAME
    # `_metric`/`_cov` over each bin mask and plot the metric as a line over an ORDINAL
    # category axis — one equally-spaced, labelled slot per bin ("0–100" … "n=…"), the
    # labelled-category style of the subclass bar figures above. Each var pairs with
    # its own target (t↦t, s↦s, pos↦pos; f_i/f_t↦t); override with `bin_targets`.
    # ------------------------------------------------------------------
    # Round candidate edges spanning [0,10000]; fine young, uniform 500-wide old. Each
    # is snapped to BRACKET the observed [min,max] so labels stay round and no empty
    # tail is drawn. `t` is np.ceil'd first (float sim generation-times → integer ages).
    _DEFAULT_EDGES = {
        "t":  [0, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500,
               5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000],
        "s":  [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05],
        "f_i": [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0],
        "f_t": [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0],
        "pos": [.1, .2, .3, .4, .5, .6, .7, .8, .9],
    }

    def _edges_for(var, v):
        """Round edges (default/override) that BRACKET the observed [min,max]; ends
        snap to a default edge when one exists (clean labels), else the data bound."""
        e = np.asarray((bin_edges or {}).get(var, _DEFAULT_EDGES.get(var, [])), float)
        vmin, vmax = float(np.nanmin(v)), float(np.nanmax(v))
        if vmax <= vmin:
            return None
        if e.size == 0:                              # unknown var → uniform fallback
            return np.linspace(vmin, vmax, 11)
        interior = e[(e > vmin) & (e < vmax)]
        lows, highs = e[e <= vmin], e[e >= vmax]
        lo = float(lows.max()) if lows.size else vmin
        hi = float(highs.min()) if highs.size else vmax
        return np.concatenate(([lo], interior, [hi]))

    def _binned_groups(v, edges):
        """Greedy left→right merge of fine [edges] bins until each holds ≥ bin_min_n.

        Returns (masks, counts, edges_used); the last bin is right-closed so the
        maximum value is included. Merged bins simply span a wider range in the label.
        """
        masks, counts, bounds = [], [], []
        cur_mask = None
        cur_lo = None
        cur_n = 0
        nfine = len(edges) - 1
        for i in range(nfine):
            a, b = edges[i], edges[i + 1]
            m = (v >= a) & (v <= b) if i == nfine - 1 else (v >= a) & (v < b)
            cur_mask = m if cur_mask is None else (cur_mask | m)
            cur_lo = a if cur_lo is None else cur_lo
            cur_n += int(m.sum())
            if cur_n >= bin_min_n or i == nfine - 1:
                if cur_n < bin_min_n and masks:       # trailing thin bin → merge left
                    masks[-1] = masks[-1] | cur_mask
                    counts[-1] += cur_n
                    bounds[-1] = (bounds[-1][0], b)
                else:
                    masks.append(cur_mask); counts.append(cur_n)
                    bounds.append((float(cur_lo), float(b)))
                cur_mask = None; cur_lo = None; cur_n = 0
        edges_used = np.array([bounds[0][0]] + [b for _, b in bounds])
        return masks, np.array(counts), edges_used

    # metric key -> (reduce-name understood by `_metric`, axis label, filename token)
    bin_metrics = [("mae", "mae", "MAE", "mae"),
                   ("rmse", "rmse", "RMSE", "rmse"),
                   ("width", "sharpness", f"{A90}% interval width", "width"),
                   ("coverage", "coverage", f"{A90}% coverage", "coverage")]

    for var in bin_by:
        if var not in df.columns:
            logger.warning("bin_by=%r not a column — skipped", var)
            continue
        v = df[var].to_numpy().astype(float)
        if var in (bin_ceil or ()):          # float sim generation-times → integer ages
            v = np.ceil(v)
        ok = np.isfinite(v)
        if ok.sum() < bin_min_n:
            logger.warning("bin_by=%r has < bin_min_n finite rows — skipped", var)
            continue
        edges = _edges_for(var, v[ok])
        if edges is None or len(edges) < 2:
            logger.warning("bin_by=%r degenerate range — skipped", var)
            continue
        masks, counts, edges_used = _binned_groups(np.where(ok, v, np.nan), edges)
        if len(masks) < 2:
            logger.warning("bin_by=%r collapsed to one bin — skipped", var)
            continue

        # ordinal category axis: one slot per bin, tick label = range only (sample size
        # n is the histogram strip under every metric — same bins, same order)
        bounds = list(zip(edges_used[:-1], edges_used[1:]))
        cats = [f"{lo:g}–{hi:g}" for lo, hi in bounds]
        xidx = np.arange(len(masks))

        want = (bin_targets or {}).get(var, [var] if var in tgs else ["t"])
        show = [nm for nm in want if nm in tgs]
        xlab = {"t": "t (generations)", "s": s_label}.get(var, var)

        for nm in show:
            col = palette.get(nm, "#3a7ca5")
            for mkey, reduce_nm, ylab, ftok in bin_metrics:
                vals = np.array([_metric(m, nm, reduce_nm) for m in masks])
                add_hist = True                     # n-strip on every metric, for consistency
                if add_hist:
                    fig, (a, an) = plt.subplots(
                        2, 1, sharex=True,
                        figsize=(max(6.0, 0.95 * len(cats)), 5.4),
                        gridspec_kw=dict(height_ratios=[1, 0.32], hspace=0.12))
                else:
                    fig, a = plt.subplots(figsize=(max(6.0, 0.95 * len(cats)), 4.8))
                    an = None
                if mkey == "coverage":
                    p = np.clip(vals, 0, 1)
                    se = np.sqrt(p * (1 - p) / np.maximum(counts, 1))
                    a.errorbar(xidx, vals, yerr=se, fmt="-o", ms=5, lw=1.6,
                               color=col, capsize=3)
                    a.axhline(A90 / 100, ls="--", c="#d1495b", lw=1.2,
                              label=f"{A90}% nominal")
                    a.set_ylim(min(0.6, float(np.nanmin(vals - se)) - 0.02), 1.0)
                    a.legend(fontsize=8, loc="lower right")
                else:
                    a.plot(xidx, vals, "-o", ms=5, lw=1.6, color=col)
                if mkey == "width" and nm == "t":
                    a.axhline(abcrf_t_width90, ls="--", c="k", lw=1.2,
                              label=f"ABC-RF {abcrf_t_width90:.0f}")
                    a.legend(fontsize=8, loc="upper left")
                a.set_ylabel(f"{labels.get(nm, nm)} — {ylab}")
                a.set_title(f"{nm}: {ylab} vs true {var}")
                a.grid(alpha=0.25, axis="y")
                xaxis_ax = an if add_hist else a    # labels go on the bottom-most axis
                if add_hist:
                    an.bar(xidx, counts, width=0.85, color="#adb5bd")  # same bins
                    an.set_ylabel("n loci\nper bin", fontsize=8)
                    an.grid(alpha=0.25, axis="y")
                    a.tick_params(axis="x", labelbottom=False)
                xaxis_ax.set_xticks(xidx)
                xaxis_ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=8)
                xaxis_ax.set_xlabel(f"true {xlab}  (bin)")
                fname = (f"fig_vs_{var}_{ftok}.png" if nm == var
                         else f"fig_vs_{var}_{nm}_{ftok}.png")
                path = os.path.join(outdir, fname)
                fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
                logger.info("evaluation figure -> %s", path)


def plot_movement_b(
    geometry: dict | None = None,
    architecture: dict | None = None,
    budget: dict | None = None,
    outdir: str = ".",
    *,
    target: str = "t",
    alpha: int = 90,
    abcrf_width: float | None = 2372.0,
    abcrf_mae: float | None = None,
    unit: str = "generations",
    knee: int | None = None,
    fname: str = "fig_movement_b.png",
) -> None:
    """Publication Figure 2 — Movement B: what drives the age-interval width.

    Every configuration is conformalized to nominal coverage, so **width is the single
    objective** and coverage is only a validity check. Two rows (width, MAE) × up to
    three columns (geometry, architecture, budget), sharing a y-scale per row and an
    ABC-RF reference line, so the reader scans the causal chain representation → network
    → data in one glance and sees every NPE config beat the classical baseline.

    Inputs — each configuration is one labeled :func:`predict_npe` output (DataFrame or
    parquet path) that keeps the true ``target`` column plus ``{target}_point`` and
    ``{target}_lo{alpha}``/``_hi{alpha}``:

    * ``geometry`` / ``architecture`` — ordered ``{label: predictions}`` for the discrete
      sweeps (drawn as sorted horizontal lollipops, tightest first, winner starred,
      cov{alpha} annotated on each mark).
    * ``budget`` — ``{series_label: {n_sims: predictions}}`` for the scaling curve (drawn
      as width/MAE vs budget lines, one per series, knee marked). Two series show the
      geometry×budget interaction (e.g. ``ms·conv`` vs ``log_scale·conv``).

    Any column may be omitted (``None``) — the figure narrows to the panels provided.
    """
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    a = (100 - alpha) / 200.0  # unused directly; intervals come from stored columns
    WIN, BASE, REF = "#e76f51", "#3a7ca5", "#6c757d"

    def _reduce(pred):
        df = pl.read_parquet(pred) if isinstance(pred, str) else pred
        y = df[target].to_numpy().astype(float)
        lo = df[f"{target}_lo{alpha}"].to_numpy().astype(float)
        hi = df[f"{target}_hi{alpha}"].to_numpy().astype(float)
        pt = df[f"{target}_point"].to_numpy().astype(float)
        return (float(np.mean(hi - lo)),                         # width
                float(np.mean(np.abs(pt - y))),                  # MAE
                float(np.mean((y >= lo) & (y <= hi))))           # coverage

    # ---- reduce every configuration to (width, MAE, cov) --------------------
    disc = []   # (col_title, subtitle, [(label, w, mae, cov), ...])  sorted by width
    if geometry is not None:
        rows = [(k, *_reduce(v)) for k, v in geometry.items()]
        rows.sort(key=lambda r: r[1])
        disc.append(("geometry", "conv · 100k", rows))
    if architecture is not None:
        rows = [(k, *_reduce(v)) for k, v in architecture.items()]
        rows.sort(key=lambda r: r[1])
        disc.append(("architecture", "ms · 100k", rows))
    bud = None
    if budget is not None:
        bud = {}
        for series, by_n in budget.items():
            pts = sorted((int(n), *_reduce(v)) for n, v in by_n.items())
            bud[series] = pts     # list of (n, w, mae, cov)

    ncol = len(disc) + (1 if bud else 0)
    if ncol == 0:
        logger.warning("plot_movement_b: nothing to plot"); return
    fig, ax = plt.subplots(2, ncol, squeeze=False, figsize=(4.8 * ncol, 7.2))

    def _lollipop(axw, axm, title, subtitle, rows):
        labels = [r[0] for r in rows]
        ws = np.array([r[1] for r in rows]); maes = np.array([r[2] for r in rows])
        covs = [r[3] for r in rows]
        y = np.arange(len(rows))[::-1]                 # tightest at TOP
        win = int(np.argmin(ws))
        cols = [WIN if i == win else BASE for i in range(len(rows))]
        # width row
        axw.hlines(y, 0, ws, color=cols, lw=2.2, zorder=1)
        axw.scatter(ws, y, s=70, color=cols, zorder=2)
        for i, (w, c) in enumerate(zip(ws, covs)):
            axw.annotate(f"cov {c:.2f}", (w, y[i]), textcoords="offset points",
                         xytext=(8, 0), va="center", fontsize=7, color="#333")
        axw.scatter(ws[win], y[win], marker="*", s=180, color=WIN, zorder=3)
        if abcrf_width:
            axw.axvline(abcrf_width, ls="--", color=REF, lw=1.2)
            axw.annotate(f"ABC-RF {abcrf_width:.0f}", (abcrf_width, y.max()),
                         textcoords="offset points", xytext=(-4, 6), ha="right",
                         fontsize=7, color=REF)
        axw.set_yticks(y); axw.set_yticklabels(labels)
        axw.set_xlim(0, max(float(ws.max()) * 1.25,
                            (abcrf_width or 0) * 1.08))
        axw.set_xlabel(f"{target} {alpha}% interval width ({unit})")
        axw.set_title(f"{title}\n({subtitle})", fontsize=10)
        axw.grid(alpha=0.25, axis="x")
        # MAE row, SAME config order
        axm.hlines(y, 0, maes, color=cols, lw=2.2, zorder=1)
        axm.scatter(maes, y, s=70, color=cols, zorder=2)
        axm.scatter(maes[win], y[win], marker="*", s=180, color=WIN, zorder=3)
        if abcrf_mae:
            axm.axvline(abcrf_mae, ls="--", color=REF, lw=1.2)
        axm.set_yticks(y); axm.set_yticklabels(labels)
        axm.set_xlim(0, float(maes.max()) * 1.25)
        axm.set_xlabel(f"{target} MAE ({unit})")
        axm.grid(alpha=0.25, axis="x")

    col = 0
    for title, subtitle, rows in disc:
        _lollipop(ax[0][col], ax[1][col], title, subtitle, rows)
        col += 1

    if bud:
        axw, axm = ax[0][col], ax[1][col]
        for i, (series, pts) in enumerate(bud.items()):
            ns = np.array([p[0] for p in pts])
            ws = np.array([p[1] for p in pts]); maes = np.array([p[2] for p in pts])
            ls = "-" if i == 0 else "--"
            c = BASE if i == 0 else "#8a6d3b"
            axw.plot(ns, ws, ls + "o", color=c, lw=1.8, ms=5, label=series)
            axm.plot(ns, maes, ls + "o", color=c, lw=1.8, ms=5, label=series)
        for axx in (axw, axm):
            axx.set_xscale("log")
            axx.set_xlabel("training simulations")
            axx.grid(alpha=0.25)
            if knee:
                axx.axvline(knee, ls=":", color=WIN, lw=1.4)
                axx.annotate(f"knee ~{knee//1000}k", (knee, axx.get_ylim()[1]),
                             textcoords="offset points", xytext=(4, -10),
                             fontsize=7, color=WIN)
        if abcrf_width:
            axw.axhline(abcrf_width, ls="--", color=REF, lw=1.2)
            axw.annotate(f"ABC-RF {abcrf_width:.0f}", (ns.min(), abcrf_width),
                         textcoords="offset points", xytext=(2, 4), fontsize=7,
                         color=REF)
        axw.set_ylabel(f"{target} {alpha}% interval width ({unit})")
        axm.set_ylabel(f"{target} MAE ({unit})")
        axw.set_title("training budget\n(scaling curve)", fontsize=10)
        axw.legend(fontsize=7); axm.legend(fontsize=7)

    ax[0][0].set_ylabel("width  ·  lower = tighter")
    ax[1][0].set_ylabel("MAE  ·  lower = more accurate")
    fig.suptitle("Movement B — what drives the age-interval width "
                 "(coverage pinned to nominal in every cell)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("movement-B figure -> %s", path)


def plot_grid(
    grid: dict,
    scaling: dict | None = None,
    outdir: str = ".",
    *,
    target: str = "t",
    alpha: int = 90,
    unit: str | None = None,
    knee: int | None = None,
    winner: tuple | None = None,
    geom_order: tuple | None = None,
    emb_order: tuple = ("fc", "cnn", "conv"),
    fname: str = "fig_movement_b.png",
) -> None:
    """Publication Figure 2 (full-grid form) — Movement B: what drives age-interval width.

    With the full geometry x embedding grid in hand, the joint optimum is *shown*, not
    inferred from a fixed-point argument. Every cell is conformalized to nominal
    coverage, so **width is the single objective** (MAE is the accuracy companion,
    coverage a validity check annotated on the heatmaps).

    Layout (2 x 2):
        [ width heatmap  (geometry x embedding) ] [ MAE heatmap (geometry x embedding) ]
        [ width vs budget @ winning embedding    ] [ MAE vs budget @ winning embedding  ]

    Inputs — each configuration is one labeled :func:`predict_npe` output (DataFrame or
    parquet path) keeping true ``target`` + ``{target}_point`` + ``{target}_lo/hi{alpha}``:

    * ``grid`` — ``{geometry: {embedding: predictions}}`` at one fixed budget: the
      geometry x embedding heatmaps. Lower width = tighter; the argmin cell is starred.
    * ``scaling`` — ``{geometry: {n_sims: predictions}}`` at the winning embedding: the
      width/MAE-vs-budget facets (one line per geometry, ``winner`` bolded, ``knee``
      marked). Omit to draw the heatmaps only.

    ``winner`` = ``(geometry, embedding)`` to star/bold; defaults to the argmin-width cell.
    """
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    WIN, REF = "#e76f51", "#6c757d"

    # unit label + numeric format adapt to the target's magnitude: t/pos are large-ish
    # naturals, s ~ [0.01, 0.05] would round to 0 under an integer format.
    if unit is None:
        unit = {"t": "generations", "pos": "relative", "s": ""}.get(target, "")
    ulab = f" ({unit})" if unit else ""

    def _fmt(v):
        if not np.isfinite(v):
            return ""
        av = abs(v)
        if av >= 100:
            return f"{v:.0f}"
        if av >= 1:
            return f"{v:.1f}"
        return f"{v:.3g}"          # small s widths/errors: 0.0352, 0.00912

    def _reduce(pred):
        df = pl.read_parquet(pred) if isinstance(pred, str) else pred
        y = df[target].to_numpy().astype(float)
        lo = df[f"{target}_lo{alpha}"].to_numpy().astype(float)
        hi = df[f"{target}_hi{alpha}"].to_numpy().astype(float)
        pt = df[f"{target}_point"].to_numpy().astype(float)
        return (float(np.mean(hi - lo)), float(np.mean(np.abs(pt - y))),
                float(np.mean((y >= lo) & (y <= hi))))

    geoms = list(geom_order) if geom_order else list(grid.keys())
    embs = [e for e in emb_order if any(e in grid.get(g, {}) for g in geoms)]
    nG, nE = len(geoms), len(embs)
    W = np.full((nG, nE), np.nan)
    M = np.full((nG, nE), np.nan)
    Cov = np.full((nG, nE), np.nan)
    for i, g in enumerate(geoms):
        for j, e in enumerate(embs):
            if e in grid.get(g, {}):
                W[i, j], M[i, j], Cov[i, j] = _reduce(grid[g][e])

    if winner is None:                                   # argmin width cell
        wi, wj = np.unravel_index(np.nanargmin(W), W.shape)
        winner = (geoms[wi], embs[wj])
    wg, we = winner
    wi, wj = geoms.index(wg), embs.index(we)

    have_scaling = bool(scaling)
    nrow = 2 if have_scaling else 1
    fig, ax = plt.subplots(nrow, 2, squeeze=False,
                           figsize=(5.2 * 2, 4.4 * nrow))

    def _heat(a, Z, title, fmt):
        im = a.imshow(Z, cmap="YlOrRd", aspect="auto")
        a.set_xticks(range(nE)); a.set_xticklabels(embs)
        a.set_yticks(range(nG)); a.set_yticklabels(geoms)
        a.set_title(title, fontsize=11)
        vmin, vmax = np.nanmin(Z), np.nanmax(Z)
        for i in range(nG):
            for j in range(nE):
                if not np.isfinite(Z[i, j]):
                    continue
                norm = (Z[i, j] - vmin) / (vmax - vmin + 1e-12)
                tc = "white" if norm > 0.6 else "black"
                lab = fmt(Z[i, j])
                if Z is W:                               # add coverage under width
                    lab += f"\ncov {Cov[i, j]:.2f}"
                a.text(j, i, lab, ha="center", va="center", fontsize=8, color=tc)
                if (i, j) == (wi, wj):                   # star the winner
                    a.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                              edgecolor=WIN, lw=2.5))
                    a.text(j, i - 0.36, "★", ha="center", va="center",
                           color=WIN, fontsize=11)
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)

    _heat(ax[0][0], W, f"{target} {alpha}% interval width{ulab}", _fmt)
    _heat(ax[0][1], M, f"{target} MAE{ulab}", _fmt)

    if have_scaling:
        cmap = plt.get_cmap("tab10")
        gcolors = {g: cmap(i % 10) for i, g in enumerate(geoms)}
        for col, which in ((0, "width"), (1, "mae")):
            a = ax[1][col]
            for g in geoms:
                if g not in scaling:
                    continue
                pts = sorted((int(n), *_reduce(v)) for n, v in scaling[g].items())
                ns = np.array([p[0] for p in pts])
                yv = np.array([p[1] if which == "width" else p[2] for p in pts])
                bold = (g == wg)
                a.plot(ns, yv, "-o", color=gcolors[g], lw=2.6 if bold else 1.4,
                       ms=6 if bold else 4, zorder=3 if bold else 2,
                       label=(g + "  ★" if bold else g))
            a.set_xscale("log")
            a.set_xlabel("training simulations")
            a.set_ylabel(f"{target} {alpha}% width{ulab}" if which == "width"
                         else f"{target} MAE{ulab}")
            a.grid(alpha=0.25)
            if knee:
                a.axvline(knee, ls=":", color=REF, lw=1.4)
                a.annotate(f"knee ~{knee // 1000}k", (knee, a.get_ylim()[1]),
                           textcoords="offset points", xytext=(4, -10),
                           fontsize=7, color=REF)
            a.set_title(f"scaling @ {we}  (width vs budget)" if which == "width"
                        else f"scaling @ {we}  (MAE vs budget)", fontsize=10)
            a.legend(fontsize=7, title="geometry")

    fig.suptitle("Movement B — geometry × embedding grid and the scaling surface "
                 "(coverage pinned to nominal in every cell)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info("movement-B grid figure -> %s", path)

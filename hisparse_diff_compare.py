#!/usr/bin/env python
"""D1: compare eager vs graph hisparse diff-dump snapshots.

Usage:
    python hisparse_diff_compare.py --eager-dir /root/hisparse_dump/eager \
                                    --graph-dir /root/hisparse_dump/graph

Snapshots are written by SelectiveHiSparseCoordinator.dump_diff_snapshot
(env SGLANG_SELECTIVE_DIFF_DUMP=1). Each file holds per-selected-layer
capture tensors over the first N forwards/replays of a run:

    locs     [n_layers, T, K] int64  — gather plan (exact compare)
    valid    [n_layers, T]    int64  — per-token valid count (exact)
    stg_pre  [n_layers, T]    int64  — staging byte-sum BEFORE patch (exact)
    stg_post [n_layers, T]    int64  — staging byte-sum AFTER patch (exact)
    q        [n_layers, T]    float32 — q_nope rowsum per token (tolerance)
    out      [n_layers, T]    float32 — SFA output rowsum per token (tolerance)

Steps of the two runs are NOT index-aligned (warmup replay counts differ),
so we content-align on the FIRST selected layer's q vector: once
divergence starts, upstream q differs and alignment stops — which is
itself the answer (first divergence is upstream of layer 0's attention).

Output: first-divergence localization (step, layer, field, tokens) +
per-layer divergence counts across aligned steps.
"""

import argparse
import glob
import os
import re
from collections import Counter

import torch

FIELD_TOL = {  # relative tolerance; int fields compared exactly
    "q": 5e-2,      # bf16 rowsums; batch padding changes reduction order
    "out": 5e-2,
    "locs": None,   # exact
    "valid": None,  # exact
    "stg_pre": None,   # exact
    "stg_post": None,  # exact
}
FIELDS = list(FIELD_TOL.keys())


def load_dir(d):
    files = glob.glob(os.path.join(d, "*.pt"))
    by_dev = {}
    for f in files:
        m = re.match(r".*_dev(\d+)_step(\d+)\.pt", os.path.basename(f))
        if not m:
            continue
        by_dev.setdefault(int(m.group(1)), []).append(
            (int(m.group(2)), f)
        )
    if not by_dev:
        raise SystemExit(f"no snapshots found under {d}")
    # pick the device with the most files (the active DP rank)
    dev, lst = max(by_dev.items(), key=lambda kv: len(kv[1]))
    lst.sort()
    snaps = [torch.load(f, map_location="cpu", weights_only=False)
             for _, f in lst]
    return dev, snaps


def rel_diff(a, b):
    """Per-row relative difference of two 1-D tensors."""
    denom = a.abs().clamp(min=1e-6)
    return ((a - b).abs() / denom)


def first_bad_idx(a, b, tol):
    if tol is None:
        bad = (a != b)
    else:
        bad = rel_diff(a.float(), b.float()) > tol
    idx = torch.nonzero(bad, as_tuple=False).flatten()
    return idx.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eager-dir", required=True)
    ap.add_argument("--graph-dir", required=True)
    args = ap.parse_args()

    edev, esnaps = load_dir(args.eager_dir)
    gdev, gsnaps = load_dir(args.graph_dir)
    print(f"eager: dev{edev}, {len(esnaps)} steps | "
          f"graph: dev{gdev}, {len(gsnaps)} steps")
    if edev != gdev:
        print(f"WARNING: active devices differ ({edev} vs {gdev}); "
              f"requests may not correspond — results unreliable.")

    layers = esnaps[0]["layers"]
    n_layers = len(layers)

    # ---- content alignment on layer-0 q ----
    pairs = []
    gi = 0
    for ei, e in enumerate(esnaps):
        t = min(e["T"], gsnaps[min(gi, len(gsnaps) - 1)]["T"]) if gsnaps else 0
        matched = -1
        while gi < len(gsnaps):
            g = gsnaps[gi]
            t = min(e["T"], g["T"])
            qe = e["q"][0, :t]
            qg = g["q"][0, :t]
            if bool((rel_diff(qe, qg) <= 5e-2).all()):
                matched = gi
                gi += 1
                break
            gi += 1
        if matched < 0:
            print(f"\n[ALIGN] eager step {e['step']}: no matching graph step "
                  f"remains — upstream divergence before layer {layers[0]} "
                  f"(or runs processed different requests). Stopping.")
            break
        pairs.append((e, gsnaps[matched]))
    print(f"aligned step pairs: {len(pairs)}")
    if not pairs:
        return

    # ---- per-pair, per-layer, per-field comparison ----
    first_div = None          # (pair_idx, layer_idx, field, bad_tokens)
    layer_bad_counts = Counter()   # (layer_idx, field) -> n_tokens
    per_pair_summary = []

    for pi, (e, g) in enumerate(pairs):
        t = min(e["T"], g["T"])
        row = {"pair": pi, "eager_step": e["step"], "graph_step": g["step"],
               "bad": []}
        for li in range(n_layers):
            for field in FIELDS:
                ea = e[field][li, :t]
                ga = g[field][li, :t]
                bad = first_bad_idx(ea, ga, FIELD_TOL[field])
                if bad:
                    row["bad"].append((li, field, len(bad), bad[:5]))
                    layer_bad_counts[(li, field)] += len(bad)
                    if first_div is None:
                        first_div = (pi, li, field, bad)
        per_pair_summary.append(row)

    # ---- report ----
    print("\n=== per-pair divergence ===")
    for row in per_pair_summary:
        if not row["bad"]:
            print(f"pair {row['pair']:3d} (e{row['eager_step']:03d}/"
                  f"g{row['graph_step']:03d}): MATCH")
        else:
            desc = ", ".join(
                f"L{layers[li]}({field}):{cnt}tok"
                for li, field, cnt, _ in row["bad"]
            )
            print(f"pair {row['pair']:3d} (e{row['eager_step']:03d}/"
                  f"g{row['graph_step']:03d}): {desc}")

    print("\n=== first divergence ===")
    if first_div is None:
        print("NONE — all aligned pairs match on every field/layer.")
        print("(If precision still differs, the poison is downstream of "
              "SFA output or in non-selected layers.)")
        return
    pi, li, field, bad = first_div
    e, g = pairs[pi]
    t = min(e["T"], g["T"])
    print(f"pair {pi} (eager step {e['step']}, graph step {g['step']}), "
          f"layer {layers[li]} (idx {li}), field '{field}', "
          f"{len(bad)} bad token(s): {bad[:10]}")
    ea = e[field][li, :t]
    ga = g[field][li, :t]
    for tok in bad[:5]:
        print(f"  token {tok}: eager={ea[tok].item():.6g} "
              f"graph={ga[tok].item():.6g}")
    print("""
chain interpretation (first divergent field at the first divergent layer):
  q        -> divergence is UPSTREAM (model state of a previous layer,
             hisparse-independent) — look at earlier layers' 'out'
  locs     -> topk/anchor compute diverged (upstream propagation)
  stg_pre  -> H2D DMA delivered different bytes with identical plan
             -> DMA data path bug (the smoking gun)
  stg_post -> current-patch wrote different content (scratch/pack path)
  out      -> inputs identical but SFA computed differently
             -> SFA/unpack workspace issue""")

    print("\n=== per-layer divergence token counts ===")
    for (li, field), cnt in sorted(layer_bad_counts.items()):
        print(f"layer {layers[li]:3d} (idx {li:2d}) {field:9s}: {cnt}")


if __name__ == "__main__":
    main()

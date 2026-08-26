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
    "pkg": None,    # exact — publish-landed scratch rowsum (pack path)
    "crow": None,   # exact — current_source_row >= 0 count (patch plan)
    "allv": None,   # exact — all_valid_mask count (SFA valid count)
    "pub": None,    # exact — publish-time packed_kv rowsum (quant output)
    "kin": 1e-4,    # pre-quant cache_k rowsum (f32; deterministic kernel
    "kinv": 1e-4,   #  + same shapes => identical inputs give identical sums)
}
FIELDS = list(FIELD_TOL.keys())


def load_dir(d, dev=None):
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
    if dev is None:
        # pick the device with the most files (the active DP rank)
        dev, lst = max(by_dev.items(), key=lambda kv: len(kv[1]))
    else:
        if dev not in by_dev:
            raise SystemExit(
                f"dev {dev} not present under {d} "
                f"(found: {sorted(by_dev)})"
            )
        lst = by_dev[dev]
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
    ap.add_argument("--dev", type=int, default=None,
                    help="compare this DP rank's device on both sides "
                    "(same rank => same requests => alignable beyond "
                    "warmup); default: auto-pick common dev")
    args = ap.parse_args()

    if args.dev is None:
        # Auto: enumerate devs of both dirs, prefer a COMMON dev (same DP
        # rank processes the same request stream in both runs — required
        # for step alignment beyond the warmup batch).
        def devs_of(d):
            return {int(m.group(1)) for f in glob.glob(os.path.join(d, "*.pt"))
                    if (m := re.match(r".*_dev(\d+)_step\d+\.pt",
                                      os.path.basename(f)))}
        common = devs_of(args.eager_dir) & devs_of(args.graph_dir)
        if not common:
            raise SystemExit("no common device between the two dirs")
        args.dev = min(common)  # deterministic

    edev, esnaps = load_dir(args.eager_dir, dev=args.dev)
    gdev, gsnaps = load_dir(args.graph_dir, dev=args.dev)
    print(f"eager: dev{edev}, {len(esnaps)} steps | "
          f"graph: dev{gdev}, {len(gsnaps)} steps")

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
                if field not in e or field not in g:
                    continue  # older snapshots without the field
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

    # ---- patch effect: within-run stg_post vs stg_pre ----
    # Distinguishes "graph patch never fired (stg_post==stg_pre in graph)"
    # from "patch fired but wrote different src bytes".
    print("\n=== patch effect (within-run stg_post vs stg_pre) ===")
    for pi, (e, g) in enumerate(pairs[:3]):
        t = min(e["T"], g["T"])
        for li in range(n_layers):
            e_pre, e_post = e["stg_pre"][li, :t], e["stg_post"][li, :t]
            g_pre, g_post = g["stg_pre"][li, :t], g["stg_post"][li, :t]
            e_act = int((e_post != e_pre).sum())
            g_act = int((g_post != g_pre).sum())
            g_noop = bool((g_post == g_pre).all())
            if e_act != g_act or g_noop:
                flag = "  <-- GRAPH PATCH NO-OP" if g_noop else ""
                print(f"pair {pi} L{layers[li]:3d}: eager patched {e_act}/{t}, "
                      f"graph patched {g_act}/{t}{flag}")
        # per-token table for the FIRST selected layer of this pair
        li = 0
        print(f"pair {pi} L{layers[li]} per-token (first 6 tokens):")
        for tok in range(min(6, t)):
            print(f"  tok {tok}: eager pre={e['stg_pre'][li, tok].item():>22d} "
                  f"post={e['stg_post'][li, tok].item():>22d} | "
                  f"graph pre={g['stg_pre'][li, tok].item():>22d} "
                  f"post={g['stg_post'][li, tok].item():>22d}")

    # ---- publish vs patch-view: within-run pub vs pkg (stale detection) ----
    # pub = what the quant produced at publish time; pkg = what the patch
    # saw in the scratch. Within-run mismatch = STALE scratch read.
    # Cross-run pub mismatch = quant itself produced different bytes.
    print("\n=== publish vs patch-view (pub vs pkg) ===")
    have_pub = all("pub" in s and "pkg" in s
                   for s in esnaps + gsnaps)
    if not have_pub:
        print("(old snapshots lack pub/pkg — re-dump with the new "
              "instrumentation to enable this check)")
    else:
        for pi, (e, g) in enumerate(pairs[:3]):
            t = min(e["T"], g["T"])
            any_hit = False
            for li in range(n_layers):
                e_stale = int((e["pub"][li, :t] != e["pkg"][li, :t]).sum())
                g_stale = int((g["pub"][li, :t] != g["pkg"][li, :t]).sum())
                x_pub = int((e["pub"][li, :t] != g["pub"][li, :t]).sum())
                if e_stale or g_stale or x_pub:
                    any_hit = True
                    verdict = []
                    if g_stale and not e_stale:
                        verdict.append("GRAPH STALE SCRATCH READ")
                    if x_pub:
                        verdict.append("QUANT OUTPUT DIFFERS (cross-run)")
                    print(f"pair {pi} L{layers[li]:3d}: stale eager="
                          f"{e_stale}/{t} graph={g_stale}/{t}, "
                          f"cross-run pub diff={x_pub}/{t}"
                          f"  <-- {'; '.join(verdict)}")
            if not any_hit:
                print(f"pair {pi}: pub == pkg on both runs for all layers "
                      f"(no staleness, same quant output)")
        # kin-vs-pub decision line for the first pair, first layers
        if pairs and "kin" in pairs[0][0] and "kin" in pairs[0][1]:
            e, g = pairs[0]
            t = min(e["T"], g["T"])
            kin_bad = sum(
                int((rel_diff(e["kin"][li, :t], g["kin"][li, :t])
                     > FIELD_TOL["kin"]).sum())
                for li in range(n_layers)
            )
            kinv_bad = sum(
                int((rel_diff(e["kinv"][li, :t], g["kinv"][li, :t])
                     > FIELD_TOL["kinv"]).sum())
                for li in range(n_layers)
            )
            pub_bad = sum(
                int((e["pub"][li, :t] != g["pub"][li, :t]).sum())
                for li in range(n_layers)
            )
            print(f"\n[VERDICT] pair 0: kin bad={kin_bad}, kinv bad="
                  f"{kinv_bad}, pub bad={pub_bad} (tokens over all layers)")
            if pub_bad and not kin_bad and not kinv_bad:
                print("  => inputs identical, quant output differs: "
                      "divergence INSIDE npu_dynamic_quant "
                      "(captured-replay semantics) — PRIME SUSPECT")
            elif kin_bad or kinv_bad:
                print("  => pre-quant inputs differ: divergence is "
                      "UPSTREAM of the quant (KV projection numerics)")
            elif not pub_bad:
                print("  => quant inputs and outputs both match")

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
    # q margin at the first divergent layer: is q (tolerance-matched)
    # actually near-identical or just under the 5% threshold? A large q
    # margin means upstream numerics, not hisparse, explains the diff.
    qe, qg = e["q"][li, :t], g["q"][li, :t]
    qrel = rel_diff(qe, qg)
    print(f"  q margin at this layer: max rel diff "
          f"{qrel.max().item():.4g} (tolerance 5e-2)")
    print("""
chain interpretation (first divergent field at the first divergent layer):
  q        -> divergence is UPSTREAM (model state of a previous layer,
             hisparse-independent) — look at earlier layers' 'out'
  locs     -> topk/anchor compute diverged (upstream propagation)
  pkg      -> publish-landed scratch differs: pack path
             (set_kv_buffer fp8 quant / scratch copy) — check q margin:
             small q noise + pkg diff = benign eager-vs-graph numerics
  crow     -> patch plan differs: current_source_row stale in graph
             (loc-plan capture/replay semantics)
  allv     -> SFA valid-count differs: valid_mask or current_source_row
             stale — attention over wrong row set
  stg_pre  -> H2D DMA delivered different bytes with identical plan
             -> DMA data path bug (the smoking gun)
  pub      -> fp8 pack output differs between runs at the source.
             CROSS-CHECK with kin at the same layer:
               kin match + pub diff -> divergence INSIDE the fp8 quant op
                                      (npu_dynamic_quant captured-replay
                                      semantics) — prime suspect
               kin diff             -> upstream KV projection differs
               kin absent           -> old snapshot; re-dump to decide
  kin/kinv -> pre-quant pack input (cache_k/cache_v) differs: the
             divergence predates the quant — upstream projection or
             mode-dependent matmul numerics
  stg_post -> patch wrote different content: if pkg matches, the
             where()/staging write itself; if pkg differs, upstream of it
  out      -> inputs identical but SFA computed differently
             -> SFA/unpack workspace issue""")

    print("\n=== per-layer divergence token counts ===")
    for (li, field), cnt in sorted(layer_bad_counts.items()):
        print(f"layer {layers[li]:3d} (idx {li:2d}) {field:9s}: {cnt}")


if __name__ == "__main__":
    main()

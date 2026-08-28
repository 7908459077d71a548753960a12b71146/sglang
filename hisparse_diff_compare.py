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
import json
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
    "pos": None,    # exact — forward_batch.positions at the selected layer
}
FIELDS = list(FIELD_TOL.keys())


def load_dir(d, dev=None):
    files = glob.glob(os.path.join(d, "*.pt"))
    by_dev = {}
    for f in files:
        # Only the per-layer snapshots; the round-3 state_dev*.pt bisect
        # files share the dir but hold a different schema.
        m = re.fullmatch(
            r"(eager|graph)_dev(\d+)_step(\d+)\.pt", os.path.basename(f)
        )
        if not m:
            continue
        by_dev.setdefault(int(m.group(2)), []).append(
            (int(m.group(3)), f)
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


def load_accept(d, dev):
    """Load per-step verify-round json records for *dev*.

    Current instrumentation writes accept_dev{N}_step{S}.json; the round-1
    form (accept_step{S}.json, no dev) is accepted as a fallback — with
    max-concurrency 1 only the active rank is ever non-idle, so those
    legacy files are unambiguous.
    """
    recs = {}
    legacy = {}
    for f in glob.glob(os.path.join(d, "accept_*.json")):
        base = os.path.basename(f)
        m = re.match(r"accept_dev(\d+)_step(\d+)\.json", base)
        if m:
            if int(m.group(1)) == dev:
                with open(f) as fh:
                    recs[int(m.group(2))] = json.load(fh)
            continue
        m = re.match(r"accept_step(\d+)\.json", base)
        if m:
            with open(f) as fh:
                legacy[int(m.group(1))] = json.load(fh)
    if not recs and legacy:
        recs = legacy
    return recs


def load_state(d, dev):
    """Load per-step target-forward bisect snapshots (state_dev{N}_step{S}.pt)."""
    recs = {}
    for f in glob.glob(os.path.join(d, "state_*.pt")):
        m = re.match(r"state_dev(\d+)_step(\d+)\.pt", os.path.basename(f))
        if m and int(m.group(1)) == dev:
            recs[int(m.group(2))] = torch.load(
                f, map_location="cpu", weights_only=False
            )
    return recs


def compare_state(args, first_fp_div):
    """Round-3 bisect: where inside the target forward does the first
    divergent verify round (first_fp_div, from the fingerprint section)
    actually drift?

    hidden row 0 = embedding, row i+1 = output of decoder layer i.
    rkv/rloc row = layer_id (resident KV written by set_kv_buffer).
    """
    print("\n=== target-forward bisect (state json/pt, same-numbered "
          "steps) ===")
    erecs = load_state(args.eager_dir, args.dev)
    grecs = load_state(args.graph_dir, args.dev)
    if not erecs or not grecs:
        print("(no state snapshots on one or both sides — re-dump with "
              "the round-3 instrumentation)")
        return
    common = sorted(set(erecs) & set(grecs))
    if first_fp_div is not None and first_fp_div not in common:
        print(f"(first fingerprint-divergent step {first_fp_div} has no "
              f"state snapshot; showing available steps)")

    def bisect_step(s):
        e, g = erecs[s], grecs[s]
        n = min(e["T"], g["T"], e["hidden"].shape[1], g["hidden"].shape[1])
        ids_bad = not torch.equal(e["in_ids"][:n], g["in_ids"][:n])
        pos_bad = not torch.equal(e["in_pos"][:n], g["in_pos"][:n])
        first_layer = None
        for row in range(e["hidden"].shape[0]):
            ev, gv = e["hidden"][row, :n], g["hidden"][row, :n]
            if bool((rel_diff(ev, gv) > 5e-2).any()):
                first_layer = row - 1  # -1 = embedding output
                break
        return ids_bad, pos_bad, first_layer, n

    # poison check FIRST: resident KV written by rounds BEFORE the first
    # divergent round must match, else the divergence is explained.
    pre = [s for s in common if first_fp_div is None or s < first_fp_div]
    for s in pre[:4] if pre else []:
        e, g = erecs[s], grecs[s]
        n = min(e["T"], g["T"])
        rkv_bad = [
            r - 1 for r in range(e["rkv"].shape[0])
            if bool((e["rkv"][r, :n] != g["rkv"][r, :n]).any())
        ]
        rloc_bad = [
            r - 1 for r in range(e["rloc"].shape[0])
            if bool((e["rloc"][r, :n] != g["rloc"][r, :n]).any())
        ]
        hrows_bad = [
            r - 1 for r in range(e["hidden"].shape[0])
            if bool((rel_diff(e["hidden"][r, :n], g["hidden"][r, :n])
                     > 5e-2).any())
        ]
        ids_bad = not torch.equal(e["in_ids"][:n], g["in_ids"][:n])
        tag = ""
        if ids_bad:
            tag += " IN_IDS DIFFER"
        if hrows_bad:
            tag += f" HIDDEN rows bad {hrows_bad[:8]}"
        if rkv_bad:
            tag += f" RKV layers {rkv_bad[:8]}"
        if rloc_bad:
            tag += f" LOC layers {rloc_bad[:8]}"
        last = e["hidden"].shape[0] - 1
        last_ok = not bool(
            (rel_diff(e["hidden"][last, :n], g["hidden"][last, :n])
             > 5e-2).any()
        )
        print(f"pre-divergence step {s}: "
              f"{'POISONED —' + tag if tag else 'match'} "
              f"(final-hidden row {'ok' if last_ok else 'BAD'})")

    for s in common:
        if first_fp_div is not None and s < first_fp_div:
            continue
        ids_bad, pos_bad, first_layer, n = bisect_step(s)
        draft_msg = ""
        if "dout_toks" in erecs[s] and "dout_toks" in grecs[s]:
            e, g = erecs[s], grecs[s]
            dn = min(e["T"], g["T"], len(e["dout_toks"]), len(g["dout_toks"]))
            dtoks_bad = not torch.equal(
                e["dout_toks"][:dn], g["dout_toks"][:dn]
            )
            din_bad = bool(
                (rel_diff(e["din_hidden"][:dn], g["din_hidden"][:dn]) > 5e-2)
                .any()
            ) if "din_hidden" in e and "din_hidden" in g else None
            draft_msg = (f", draft: din{'BAD' if din_bad else 'ok'} "
                         f"dout_toks{'DIFFER' if dtoks_bad else 'same'}")
        if ids_bad or pos_bad or first_layer is not None:
            print(f"step {s:3d}: inputs {'IDS DIFFER' if ids_bad else 'ids same'}, "
                  f"{'POS DIFFER' if pos_bad else 'pos same'}, "
                  f"first divergent hidden row: "
                  f"{'embedding' if first_layer == -1 else f'layer {first_layer}'} "
                  f"({n} tokens compared){draft_msg}")
        elif (first_fp_div is not None and s <= first_fp_div + 2) or (
            first_fp_div is None and s <= 3
        ):
            print(f"step {s:3d}: hidden all match ({n} tokens){draft_msg}")

    if first_fp_div is not None and first_fp_div in common:
        e, g = erecs[first_fp_div], grecs[first_fp_div]
        n = min(e["T"], g["T"])
        ids_bad, pos_bad, first_layer, _ = bisect_step(first_fp_div)

        def _nan_inf(t):
            return int(torch.isnan(t).sum()), int(torch.isinf(t).sum())

        print(f"\nNaN/Inf attribution at step {first_fp_div} "
              f"(eager, graph):")
        for fld in ("din_hidden", "dstep_logits", "dstep_hidden"):
            if fld in e and fld in g:
                en, ei = _nan_inf(e[fld])
                gn, gi = _nan_inf(g[fld])
                print(f"    {fld:14s}: eager nan/inf={en}/{ei}, "
                      f"graph nan/inf={gn}/{gi}")
        # Handoff chain for this round's bad draft input: verify(d-1)
        # output (live, at sample) -> draft-extend(d-1) output (live, at
        # gather) -> this round's chain input din (fresh gather product).
        # state file k holds dext_out from round k-1's draft-extend, so the
        # source for din(d) lives in state d (same file), hout_live for the
        # verify that fed it lives in state d-1.
        if first_fp_div is not None and (first_fp_div - 1) in erecs \
                and (first_fp_div - 1) in grecs:
            ep, gp = erecs[first_fp_div - 1], grecs[first_fp_div - 1]
            if "hout_live" in ep and "hout_live" in gp:
                rd = rel_diff(
                    ep["hout_live"][:n].float(), gp["hout_live"][:n].float()
                ).max().item()
                print(f"    hout_live(d-1) max rel {rd:.4g} "
                      f"{'<-- BAD (verify output already dirty at sample)' if rd > 5e-2 else '(clean)'}")
        if "dext_out" in e and "dext_out" in g:
            rd = rel_diff(
                e["dext_out"][:n].float(), g["dext_out"][:n].float()
            ).max().item()
            print(f"    dext_out(d)   max rel {rd:.4g} "
                  f"{'<-- BAD (draft-extend output/source dirty)' if rd > 5e-2 else '(clean)'}")
        if "din_hidden" in e and "din_hidden" in g:
            rd = rel_diff(
                e["din_hidden"][:n].float(), g["din_hidden"][:n].float()
            ).max().item()
            print(f"    din(d)        max rel {rd:.4g} "
                  f"{'<-- BAD (chain input dirty)' if rd > 5e-2 else '(clean)'}")
        if "dloc" in e and "dloc" in g:
            same = bool(torch.equal(e["dloc"][:n], g["dloc"][:n]))
            print(f"    dloc(d)       {'MATCH' if same else 'DIFFER <-- draft KV write addressing wrong (shared-state poison candidate)'}")
        if "dkvh" in e and "dkvh" in g:
            same = bool(torch.equal(e["dkvh"][:n], g["dkvh"][:n]))
            print(f"    dkvh(d)       {'MATCH' if same else 'DIFFER <-- draft KV history CONTENT poisoned (the NaN source candidate)'}")
        if "dseql" in e and "dseql" in g:
            same = bool(torch.equal(e["dseql"][:n], g["dseql"][:n]))
            print(f"    dseql(d)      {'MATCH' if same else 'DIFFER <-- draft seq_lens drift (attention metadata candidate)'}")
        # Round-8: draft-model sub-block bisect — first NaN/divergent block
        # along emb -> prev(rot matmul on the in-graph handoff read) ->
        # eh(eh_proj) -> out(final norm). NaN in the graph side at the
        # FIRST bad block pins the draft-internal poison point.
        # Round-8: draft-model sub-block bisect. Keys print in pipeline
        # order: ids/pos (graph static inputs) -> bt/topk (attention index
        # sources) -> prevraw/prev (in-graph handoff read, pre/post rot)
        # -> emb -> eh -> attn/mlp (decoder internals) -> out. The FIRST
        # graph-side NaN / divergence in this order is the poison point.
        dm_keys = [k[3:] for k in e.keys() if k.startswith("dm_")]
        if dm_keys:
            order = ("ids", "pos", "bt", "topk", "prevraw", "prev",
                     "emb", "eh", "am_q", "am_qpe", "am_tik", "am_kvlen",
                     "am_seqlens", "am_bt", "am_pgsum", "attn_raw",
                     "attn", "mlp", "out", "lmin", "lmw", "lmout")
            dm_keys.sort(key=lambda k: order.index(k)
                         if k in order else len(order))
            print("  draft-model sub-block fingerprints "
                  "(eager nan, graph nan, max rel, graph-NaN steps):")
            for k in dm_keys:
                ev, gv = e[f"dm_{k}"], g[f"dm_{k}"]
                enan = int(torch.isnan(ev.float()).sum())
                gnan = int(torch.isnan(gv.float()).sum())
                gsteps = torch.nonzero(
                    torch.isnan(gv.float()).any(dim=-1)
                ).flatten().tolist()
                rd = rel_diff(
                    ev.flatten().float(), gv.flatten().float()
                ).max().item() if ev.numel() else 0.0
                print(f"    {k:8s}: {enan}, {gnan}, {rd:.4g}, "
                      f"nan@steps{gsteps}")
        # Per-step stream-order table for the lm_head segment: the first
        # position (in replay stream order) whose GRAPH value is NaN /
        # diverges at step 0 is where the poison enters. Stream order:
        # out -> lmin -> lmw -> lmout -> [runner tail] -> dstep_logits.
        print("  step-0 chain values (eager | graph):")
        for k in ("out", "lmin", "lmw", "lmout"):
            if f"dm_{k}" in e:
                ev, gv = e[f"dm_{k}"], g[f"dm_{k}"]
                print(f"    {k:8s}: {ev[0].tolist()} | {gv[0].tolist()}")
        # Attention metadata probes: print ALL chain steps — the kvlen the
        # graph attention sees vs eager is the round-11 poison candidate.
        for k in ("am_kvlen", "am_seqlens", "am_q", "am_qpe", "am_tik",
                  "am_bt", "am_pgsum", "attn_raw"):
            if f"dm_{k}" in e and f"dm_{k}" in g:
                ev, gv = e[f"dm_{k}"], g[f"dm_{k}"]
                print(f"    {k:8s}: {ev.flatten().tolist()} | "
                      f"{gv.flatten().tolist()}")
        if "dstep_logits" in e:
            print(f"    dstep_log: {e['dstep_logits'][:2].tolist()} | "
                  f"{g['dstep_logits'][:2].tolist()}")
        if "dstep_toks" in e:
            print(f"    dstep_tok: {e['dstep_toks'][:2].tolist()} | "
                  f"{g['dstep_toks'][:2].tolist()}")
        # Root-vs-draft split: with topk=1 the verify tree is a chain and
        # in_ids[0] is the LAST ACCEPTED token of the previous round
        # (accept_lens matched there), in_ids[1:] are this round's fresh
        # draft proposals.
        print(f"\n[BISECT VERDICT] step {first_fp_div}:")
        print(f"  in_ids side-by-side (T={n}):")
        for i in range(n):
            ev, gv = int(e["in_ids"][i]), int(g["in_ids"][i])
            mark = "  <-- DIFFER" if ev != gv else ""
            role = "root(last-accepted)" if i == 0 else "draft"
            print(f"    [{i}] {role:20s} eager={ev:>8d} graph={gv:>8d}{mark}")
        n_diff_root = int(e["in_ids"][0] != g["in_ids"][0])
        n_diff_draft = sum(
            1 for i in range(1, n) if int(e["in_ids"][i]) != int(g["in_ids"][i])
        )
        if n_diff_root:
            print("  => ROOT token differs: the previous round ACCEPTED a "
                  "different token despite identical logits+accept_lens — "
                  "eagle_sample/accept-index compaction bug in graph mode")
        elif n_diff_draft:
            print(f"  => root matches, {n_diff_draft}/{n - 1} draft "
                  "proposals differ: DRAFT CHAIN divergence — "
                  "cross-check draft fields:")
            if "dout_toks" in e and "dout_toks" in g:
                dn = min(n, len(e["dout_toks"]), len(g["dout_toks"]))
                din_bad = bool(
                    (rel_diff(e["din_hidden"][:dn], g["din_hidden"][:dn])
                     > 5e-2).any()
                ) if "din_hidden" in e and "din_hidden" in g else None
                dtoks_bad = not torch.equal(
                    e["dout_toks"][:dn], g["dout_toks"][:dn]
                )
                din_topk_bad = (
                    not torch.equal(e["din_topk"][:dn], g["din_topk"][:dn])
                ) if "din_topk" in e and "din_topk" in g else None
                if din_bad and not din_topk_bad:
                    print("     din_hidden DIFFERS but din_topk (initial "
                          "proposal) matches: corruption specific to the "
                          "hidden TENSOR — graph-pool aliasing signature")
                if din_bad:
                    print("     din_hidden DIFFERS: the handoff INTO the "
                          "draft is already wrong (previous verify "
                          "round's hidden/sample state)")
                elif dtoks_bad:
                    print("     din_hidden matches + draft tokens differ: "
                          "the DRAFT FORWARD itself diverges in graph "
                          "(draft attention/KV or its metadata)")
                else:
                    print("     draft tokens match but verify in_ids "
                          "differ: build_eagle_verify_input tree "
                          "assembly bug")
            if "dstep_toks" in e and "dstep_toks" in g:
                print("  per-step inside the draft chain (step: toks, "
                      "logits rel, input-hidden rel):")
                for k in range(e["dstep_toks"].shape[0]):
                    et, gt = e["dstep_toks"][k], g["dstep_toks"][k]
                    toks_diff = not torch.equal(et, gt)
                    el, gl = e["dstep_logits"][k], g["dstep_logits"][k]
                    lrel = rel_diff(el, gl).max().item() if len(el) else 0.0
                    eh, gh = e["dstep_hidden"][k], g["dstep_hidden"][k]
                    hrel = rel_diff(eh, gh).max().item() if len(eh) else 0.0
                    flag = "  <-- FIRST DIVERGENT STEP" if (
                        toks_diff
                        and all(
                            torch.equal(e["dstep_toks"][j],
                                        g["dstep_toks"][j])
                            for j in range(k)
                        )
                    ) else ""
                    print(f"    step {k}: toks {'DIFFER' if toks_diff else 'same'},"
                          f" logits rel {lrel:.4g},"
                          f" hidden rel {hrel:.4g}{flag}")
        if ids_bad or pos_bad:
            print("  (verify inputs differ as reported above)")
        elif first_layer == -1:
            print("  => embedding output differs with identical input ids "
                  "— embedding lookup/graph input binding bug")
        elif first_layer is None:
            print("  => hidden matches through ALL layers but logits "
                  "diverge — lm_head / logits processor path")
        else:
            print(f"  => first divergence INSIDE layer {first_layer} "
                  f"(compute or its attention KV read). "
                  f"Cross-check rkv/rloc at earlier steps for this layer: "
                  "differs => poisoned resident KV written in graph; "
                  "matches => live compute/attention-metadata divergence "
                  "in that layer")


def compare_accept(args, esnaps, gsnaps):
    """Same-numbered-step comparison of the verify-round fingerprints.

    Unlike the .pt snapshots, these records need NO content alignment:
    both runs count non-idle verify rounds from 1 over the same request
    stream, and the round-1 comparison proved no warmup offset. This
    section answers where the step-2 divergence (which kills .pt
    alignment) actually lives: target forward (rowsum), sampler/verify
    tree (argmax/accept_lens with matching rowsum), or neither.
    """
    print("\n=== verify-round fingerprint (accept json, same-numbered "
          "steps) ===")
    erecs = load_accept(args.eager_dir, args.dev)
    grecs = load_accept(args.graph_dir, args.dev)
    if not erecs or not grecs:
        print("(no accept jsons found on one or both sides — re-dump)")
        return
    has_logits = all(
        "logit_rowsum" in r for r in list(erecs.values())[:1]
    ) and all("logit_rowsum" in r for r in list(grecs.values())[:1])
    if not has_logits:
        print("(accept jsons lack the logits fingerprint — server ran the "
              "round-1 instrumentation; re-run BOTH sides with the "
              "round-2 code)")
        return

    common = sorted(set(erecs) & set(grecs))
    print(f"eager steps with json: {sorted(erecs)} | graph: {sorted(grecs)}")

    def row_stats(e, g):
        """(rowsum bad?, max rel diff, argmax mismatches, n compared)"""
        er = torch.tensor(e["logit_rowsum"], dtype=torch.float64)
        gr = torch.tensor(g["logit_rowsum"], dtype=torch.float64)
        n = min(len(er), len(gr))
        rd = rel_diff(er[:n], gr[:n])
        ea = e["logit_argmax"]
        ga = g["logit_argmax"]
        amis = sum(1 for i in range(min(len(ea), len(ga)))
                   if ea[i] != ga[i])
        return bool((rd > 5e-2).any()), float(rd.max()) if n else 0.0, amis, n

    first_div = None
    for s in common:
        e, g = erecs[s], grecs[s]
        rs_bad, rs_max, amis, n = row_stats(e, g)
        acc_eq = e["accept_lens"] == g["accept_lens"]
        if rs_bad or amis or not acc_eq:
            if first_div is None:
                first_div = (s, rs_bad, amis, acc_eq)
            print(f"step {s:3d}: DIVERGE rowsum{'BAD' if rs_bad else 'ok'} "
                  f"(max rel {rs_max:.4g}), argmax mismatches {amis}/{n}, "
                  f"accept_lens {'same' if acc_eq else 'DIFFER'}")
        elif s <= 3:
            print(f"step {s:3d}: match (rowsum max rel {rs_max:.4g}, "
                  f"argmax 0/{n}, accept_lens same "
                  f"{e['accept_lens'][:4]})")
    if first_div is None:
        print(f"all {len(common)} common steps match on rowsum/argmax/"
              "accept_lens")
        print("=> the two runs compute IDENTICAL verify outputs; the .pt "
              "ALIGN failure is a scheduling/ordering artifact (different "
              "request placement or step offset) — rerun to confirm")
        return None
    s, rs_bad, amis, acc_eq = first_div
    print(f"\n[FINGERPRINT VERDICT] first divergent verify round: step {s}")
    if rs_bad:
        print("  => logit rowsum differs: TARGET FORWARD diverges in graph "
              "(non-selected layers / lm_head / logits processor). "
              "Selected-layer dumps are clean at step 1, so bisect with "
              "per-layer hidden-state rowsum probes next.")
    elif amis:
        print("  => rowsum matches but argmax differs: near-tie logits or "
              "sampler-side divergence (greedy: meaningful; sampled: check "
              "RNG seed before concluding)")
    elif not acc_eq:
        print("  => logits match but accept_lens differ: eagle_sample / "
              "verify-tree handling diverges, model forward is innocent")
    return s



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
            return {
                int(m.group(2))
                for f in glob.glob(os.path.join(d, "*.pt"))
                if (m := re.fullmatch(
                    r"(eager|graph)_dev(\d+)_step\d+\.pt",
                    os.path.basename(f),
                ))
            }
        common = devs_of(args.eager_dir) & devs_of(args.graph_dir)
        if not common:
            raise SystemExit("no common device between the two dirs")
        args.dev = min(common)  # deterministic

    edev, esnaps = load_dir(args.eager_dir, dev=args.dev)
    gdev, gsnaps = load_dir(args.graph_dir, dev=args.dev)
    print(f"eager: dev{edev}, {len(esnaps)} steps | "
          f"graph: dev{gdev}, {len(gsnaps)} steps")

    # Verify-round fingerprints first: they need no .pt-style alignment
    # (same-numbered steps compare directly) and answer where the
    # divergence that breaks .pt alignment actually lives.
    first_fp_div = compare_accept(args, esnaps, gsnaps)
    # Round-3 bisect on the first divergent round: exact layer inside the
    # target forward + resident-KV poison check on the rounds before it.
    compare_state(args, first_fp_div)

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
            # magnitude table for layer 0 (and the layer where kin first
            # matches but kinv does not, if any): absolute values decide
            # whether these are noise-level (1e-3) or real poison.
            print(f"\n  magnitude at L{layers[0]} (first {min(6, t)} tokens):")
            for fld in ("pos", "kin", "kinv", "pub", "stg_post", "out"):
                if fld not in e or fld not in g:
                    continue
                ev, gv = e[fld][0, :t], g[fld][0, :t]
                if FIELD_TOL[fld] is None:
                    badm = (ev != gv)
                else:
                    badm = rel_diff(ev.float(), gv.float()) > FIELD_TOL[fld]
                nbad = int(badm.sum())
                if nbad:
                    rd = rel_diff(ev.float(), gv.float())
                    print(f"    {fld:9s}: {nbad}/{t} bad, "
                          f"max rel diff {rd.max().item():.4g}")
                    for tok in torch.nonzero(badm).flatten()[:3].tolist():
                        print(f"      tok {tok}: eager={ev[tok].item():.6g} "
                              f"graph={gv[tok].item():.6g} "
                              f"(rel {rd[tok].item():.4g})")
                else:
                    rd = rel_diff(ev.float(), gv.float()) \
                        if FIELD_TOL[fld] is not None else None
                    extra = f", max rel diff {rd.max().item():.4g}" if rd is not None else ""
                    print(f"    {fld:9s}: 0/{t} bad{extra}")

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

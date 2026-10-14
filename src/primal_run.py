"""Primal track (sprint 2026-10-14): checked UPPER bounds from packingsolver, both cost variants.

packingsolver 0.1.1183 (module packingsolver.rectangle) is run in OptimizationMode.NotAnytimeSequential
(measured: exactly 1 CPU thread; the Anytime mode uses 3-4 threads) with small tree-search queue
sizes so that a solution appears within a few seconds even at n = 100. The full placement is
extracted through Solution.bin(i) -> SolutionBin(bin_type_id, copies, items[SolutionItem(item_type_id,
x, y, rotate)]). Every raw bin line is added as its own bin type (no dedupe), so bin_type_id IS the
0-based raw-file line index; every item is its own item type (copies=1), so item_type_id is the raw
item index.

Each packing is verified internally, written to results/solutions/sprint/<name>_<variant>_<method>_<tag>_c<cost>.json
(ledger.sol_relpath), re-checked by the independent src/check_solution.py
(subprocess) and recorded in the ledger (method 'packingsolver'). The packing is also re-costed in
the OTHER variant (a packing is feasible for both; only the cost differs) and recorded if it beats
what the ledger has for that variant.

usage:
  python src/primal_run.py --variant weighted --tl 2.5 --tag pass1 [--names MB_C1_1,...] [--q 16 --it 4]
  resumable: (name, variant, tag) already in the ledger (method packingsolver, note tag=<tag>) is skipped.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import packingsolver.rectangle as psr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ledger  # noqa: E402
import vsbpp_io as io  # noqa: E402

ROOT = io.ROOT
SOLDIR = os.path.join(ROOT, 'results', 'solutions', 'sprint')
METHOD = 'packingsolver'


def read_raw(nm):
    """bins = [(W, H, c_file)] in raw line order, items = [(w, h)] in raw order."""
    L = open(io.instance_path(nm)).read().split('\n')
    n, m, _ = map(int, L[0].split())
    bins = [tuple(map(int, L[1 + t].split())) for t in range(m)]
    items = [tuple(map(int, L[1 + m + i].split()))[1:3] for i in range(n)]
    return [(b[0], b[1], b[3]) for b in bins], items


def cost_of(place, bins, variant):
    used = {(p[0], p[1]) for p in place}
    return sum((bins[t][0] * bins[t][1] if variant == 'area' else bins[t][2]) for t, k in used)


def verify(place, bins, items):
    assert sorted((p[2], p[3]) for p in place) == sorted(items), 'multiset'
    per = {}
    for t, k, w, h, x, y in place:
        W, H = bins[t][0], bins[t][1]
        assert 0 <= x and 0 <= y and x + w <= W and y + h <= H, 'outside'
        per.setdefault((t, k), []).append((x, y, w, h))
    for rs in per.values():
        for a in range(len(rs)):
            for b in range(a + 1, len(rs)):
                xa, ya, wa, ha = rs[a]; xb, yb, wb, hb = rs[b]
                assert not (xa < xb + wb and xb < xa + wa and ya < yb + hb and yb < ya + ha), 'overlap'


def run_ps(nm, variant, tl, q=16, it=4):
    bins, items = read_raw(nm)
    b = psr.InstanceBuilder()
    b.set_objective(psr.Objective.VariableSizedBinPacking)
    for W, H, c in bins:
        b.add_bin_type(W, H, cost=(W * H if variant == 'area' else c), copies=len(items))
    for w, h in items:
        b.add_item_type(w, h, oriented=True, copies=1)
    inst = b.build()
    p = psr.OptimizeParameters()
    p.time_limit = tl; p.verbosity_level = 0; p.messages_to_stdout = False
    p.optimization_mode = psr.OptimizationMode.NotAnytimeSequential   # 1 thread
    p.not_anytime_tree_search_queue_size = q
    p.not_anytime_sequential_value_correction_number_of_iterations = it
    p.not_anytime_sequential_single_knapsack_subproblem_tree_search_queue_size = q
    p.not_anytime_dichotomic_search_subproblem_tree_search_queue_size = q
    p.sequential_value_correction_subproblem_tree_search_queue_size = q
    out = psr.optimize(inst, p)
    s = out.solution
    if not s.feasible() or s.number_of_items() != len(items):
        return None, bins, items
    place = []
    kcount = {}
    for i in range(s.number_of_different_bins()):
        sb = s.bin(i)
        t = sb.bin_type_id
        for _c in range(sb.copies):
            k = kcount.get(t, 0); kcount[t] = k + 1
            for si in sb.items:
                assert not si.rotate
                w, h = items[si.item_type_id]
                place.append([t, k, w, h, int(si.x), int(si.y)])
    return place, bins, items


def check_external(nm, solf, variant):
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'src', 'check_solution.py'),
                        io.instance_path(nm), solf, variant], capture_output=True, text=True)
    out = r.stdout.strip()
    if r.returncode == 0 and out.startswith('valid'):
        return int(out.split('cost')[-1].strip())
    return None


def store_and_record(nm, variant, place, bins, items, tag, tl, t_s, note):
    """Write the packing (one file per cost, see ledger.sol_relpath), check it externally, record it."""
    cost = cost_of(place, bins, variant)
    solrel = ledger.sol_relpath(nm, variant, METHOD, cost, tag)
    solf = os.path.join(ROOT, solrel)
    tmp = solf + '.tmp'
    json.dump({'name': nm, 'variant': variant, 'cost': cost, 'method': METHOD, 'tag': tag,
               'time_limit': tl, 'threads': 1, 'note': note, 'place': place}, open(tmp, 'w'))
    c_ext = check_external(nm, tmp, variant)
    ok = (c_ext == cost)
    if ok:
        os.replace(tmp, solf)
    else:
        os.remove(tmp)
    ledger.record(nm, variant, METHOD, 'FEASIBLE', ub=cost, time_s=t_s, threads=1,
                  sol_file=solrel if ok else '', checked=1 if ok else 0,
                  note=f'tag={tag} tl={tl}; {note}' + ('' if ok else f'; CHECK FAILED ext={c_ext}'))
    return cost, ok


def done_set(tag):
    s = set()
    for r in ledger.rows():
        if r['method'] == METHOD and f'tag={tag} ' in (r['note'] + ' '):
            s.add((r['name'], r['variant']))
        if r['method'] == METHOD and r['note'].startswith(f'tag={tag};'):
            s.add((r['name'], r['variant']))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, choices=['weighted', 'area'])
    ap.add_argument('--tl', type=float, default=2.5)
    ap.add_argument('--tag', default='pass1')
    ap.add_argument('--names', default='')
    ap.add_argument('--q', type=int, default=16)
    ap.add_argument('--it', type=int, default=4)
    ap.add_argument('--cross', type=int, default=1, help='also record re-costed packing in other variant')
    a = ap.parse_args()
    names = a.names.split(',') if a.names else io.all_names()
    done = done_set(a.tag)
    other = 'area' if a.variant == 'weighted' else 'weighted'
    for nm in names:
        if (nm, a.variant) in done:
            continue
        t0 = time.time()
        place, bins, items = run_ps(nm, a.variant, a.tl, a.q, a.it)
        note = f'q={a.q} it={a.it} NotAnytimeSequential'
        if place is None:   # retry with tiny queues
            place, bins, items = run_ps(nm, a.variant, a.tl, 2, 1)
            note = 'q=2 it=1 NotAnytimeSequential (retry)'
        t_s = time.time() - t0
        if place is None:
            ledger.record(nm, a.variant, METHOD, 'UNKNOWN', time_s=t_s, threads=1,
                          note=f'tag={a.tag} tl={a.tl}; no solution')
            print(nm, a.variant, 'NONE', round(t_s, 1), flush=True)
            continue
        verify(place, bins, items)
        cost, ok = store_and_record(nm, a.variant, place, bins, items, a.tag, a.tl, t_s, note)
        msg = [nm, a.variant, cost, 'checked' if ok else 'nofile', round(t_s, 1)]
        if a.cross:
            best = ledger.best().get((nm, other), {})
            c2 = cost_of(place, bins, other)
            if best.get('ub') is None or c2 < best['ub']:
                c2r, ok2 = store_and_record(nm, other, place, bins, items, a.tag + 'x', a.tl, t_s,
                                            f'packing of {a.variant} run re-costed')
                msg += ['cross', other, c2r, ok2]
        print(*msg, flush=True)


if __name__ == '__main__':
    main()

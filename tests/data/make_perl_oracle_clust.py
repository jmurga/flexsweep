"""
Regenerate tests/data/enrichment_perl_oracle_clust.json from the Enard-lab Perl scripts.

Not a test: run by hand when the synthetic inputs of tests/test_enrichment_parity.py or
the Perl scripts change.

    python tests/data/make_perl_oracle_clust.py ~/Gene_Set_Enrichment_Pipeline/Gene_Set_Enrichment_Pipeline_new

Oracles written:

* ``count_clust``: count_sweeps_script.pl with Count_sweeps ``yes`` (neighbour file from
  get_all_neighbors.pl at a given Cluster_distance, sweep file from
  simplify_zipped_sweeps.pl), on the synthetic rank table and on rank tables whose rows
  were permuted (as a genome shuffle does), for several Pop_interest values.
* ``segments_blocks``: shuffle_sweep_script_nocut.pl with its commented-out line
  ``$seg_size = int($sc_ov/$segcut)`` restored (segcut = number of blocks, as described
  in Enard & Petrov 2020), mirroring disabled (dice fixed at 0) and the segments
  printed; nothing else changed.
"""

import gzip
import json
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tests.test_enrichment_parity import GROUPS, POPS, THR, _coords_bed, _synthetic  # noqa: E402

DISTANCES = [20_000, 60_000]
SETPOPS = ["p1", "p3", "All"]
N_PERMUTED = 2
BLOCKS = [3, 7]


def _write_inputs(tmp: pathlib.Path, table, vip, sets) -> None:
    bed = _coords_bed(tmp)
    with open(bed) as fi, open(tmp / "coords.txt", "w") as fo:  # 1-based, Perl layout
        for ln in fi:
            c, s0, e, g, strand = ln.split()
            fo.write(f"{g} {c} {int(s0) + 1} {e} {strand}\n")
    with open(tmp / "valid_file.txt", "w") as fh:
        fh.write("\n".join(table[:, 0].to_list()) + "\n")
    (tmp / "VIPs").mkdir(exist_ok=True)
    (tmp / "nonVIPs").mkdir(exist_ok=True)
    with open(tmp / "VIPs" / "file_1", "w") as fh:
        fh.write("\n".join(vip) + "\n")
    with open(tmp / "nonVIPs" / "file_1", "w") as fh:
        for i, row in enumerate(sets):
            fh.write(f"sample_{i + 1} " + " ".join(row) + "\n")


def _write_sweeps(tmp: pathlib.Path, table, name: str) -> str:
    gz = tmp / f"{name}.gz"
    with gzip.open(gz, "wt") as fh:
        for row in table.iter_rows():
            fh.write(" ".join(row) + "\n")
    out = tmp / f"simple_{name}"
    subprocess.run(["perl", f"{PERL}/simplify_zipped_sweeps.pl", str(gz), str(out),
                    ",".join(map(str, THR))], check=True, cwd=tmp)
    return str(out)


def _perl_count(tmp, sweep, neighbors, setpop) -> list:
    outdir = tmp / "out" / f"{setpop}"
    outdir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["perl", f"{PERL}/count_sweeps_script.pl", str(tmp / "VIPs" / "file_1"),
         str(tmp / "nonVIPs" / "file_1"), sweep, neighbors, setpop, f"{outdir}/", "yes",
         ",".join(map(str, THR)), ",".join(POPS), ",".join(GROUPS)],
        check=True, cwd=tmp, stdout=subprocess.DEVNULL,
    )
    scopes = set(POPS) | set(GROUPS) | {"All"}
    rows = []
    for ln in open(outdir / os.path.basename(sweep)):
        p = ln.split()
        if len(p) == 8 and p[1].rstrip(":") in scopes:
            rows.append([int(p[0]), p[1].rstrip(":")] + [float(x) for x in p[2:]])
    return rows


def _perl_block_segments(tmp, table, vip, sets, n_blocks) -> list:
    src = open(f"{PERL}/shuffle_sweep_script_nocut.pl").read()
    edits = [
        ("my $seg_size = $segcut;", "my $seg_size = int($sc_ov/$segcut);"),
        ("my $dice=rand(10);", "my $dice=0;"),
        ("$dice=rand(10);", "$dice=0;"),
        ('print "Number of segments: ".$sc_segg."\\n";',
         'print "SEG ".$_."\\n" for @segments; exit;'),
    ]
    for old, new in edits:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    script = tmp / "shuffle_blocks.pl"
    script.write_text(src)
    used = sorted(set(vip) | set(np.unique(sets)[:80]))
    (tmp / "used_genes.txt").write_text("\n".join(used) + "\n")
    with open(tmp / "ranks_raw", "w") as fh:  # prefix "ranks" + sweep size "raw"
        for row in table.iter_rows():
            fh.write(" ".join(row) + "\n")
    res = subprocess.run(
        ["perl", str(script), f"{tmp}/", f"{tmp}/shuf/", str(tmp / "valid_file.txt"), "1",
         str(tmp / "coords.txt"), str(tmp / "ranks"), "raw", str(n_blocks), str(max(THR))],
        check=True, cwd=tmp, capture_output=True, text=True,
    )
    return [[int(x) for x in ln.split()[1:]] for ln in res.stdout.splitlines() if ln.startswith("SEG")]


def main() -> None:
    table, vip, sets = _synthetic()
    out = {"_provenance": {
        "source": PERL, "scripts": ["count_sweeps_script.pl", "simplify_zipped_sweeps.pl",
                                    "get_all_neighbors.pl", "shuffle_sweep_script_nocut.pl"],
        "perl": subprocess.run(["perl", "-e", "print $^V"], capture_output=True, text=True).stdout,
        "note": __doc__.strip().splitlines()[0],
    }, "count_clust": {}, "segments_blocks": {}}
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        _write_inputs(tmp, table, vip, sets)
        rng = np.random.default_rng(11)
        tables = {"real": table}
        for k in range(N_PERMUTED):
            perm = rng.permutation(table.height)
            tables[f"perm{k}"] = table.with_columns(
                [table[c].gather(perm) for c in table.columns[1:]])
            out.setdefault("permutations", {})[f"perm{k}"] = perm.tolist()
        for dist in DISTANCES:
            nb = tmp / f"neighbors_{dist}.txt"
            subprocess.run(["perl", f"{PERL}/get_all_neighbors.pl", str(tmp / "coords.txt"),
                            str(nb), str(dist)], check=True, cwd=tmp)
            for name, tab in tables.items():
                sweep = _write_sweeps(tmp, tab, f"{name}_{dist}")
                for setpop in SETPOPS:
                    out["count_clust"][f"{dist}|{name}|{setpop}"] = _perl_count(tmp, sweep, str(nb), setpop)
        for nbk in BLOCKS:
            out["segments_blocks"][str(nbk)] = _perl_block_segments(tmp, table, vip, sets, nbk)
    path = REPO / "tests" / "data" / "enrichment_perl_oracle_clust.json"
    json.dump(out, open(path, "w"))
    print("wrote", path, {k: len(v) for k, v in out["count_clust"].items()})


if __name__ == "__main__":
    PERL = os.path.expanduser(sys.argv[1]).rstrip("/")
    main()

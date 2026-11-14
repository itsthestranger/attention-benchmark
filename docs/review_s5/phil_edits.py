"""What the cpu tier says after an edit the runbook asks Phil to commit. Applies, runs the whole tier, restores.

Written by the reviewer of session 5 (docs/review_s5.md, H1). The build loop changed two things
afterwards: the worktree is the checkout this file is in, not the reviewer's own, and case X4 was
added (SWA's opt path, RUNBOOK §1.2). Run it from a clean checkout whenever a test touches a file
Phil edits during the campaign:

    python docs/review_s5/phil_edits.py X1 X2 X3 X4

Each case takes one run of the cpu tier, about six minutes. Every case must end with 0 failed.
"""
import subprocess, sys, os
from pathlib import Path
WT = Path(__file__).resolve().parents[2]
env = dict(os.environ, OMP_NUM_THREADS="1")
CASES = {
 "X1 commitments seed_count 5 (RUNBOOK 3.4)": [("configs/commitments.yaml", "\nseed_count: null", "\nseed_count: 5")],
 "X2 recipe frozen (RUNBOOK 2.2) and data hashes set (RUNBOOK 2.1)": [
    ("configs/recipe.yaml", "  micro_batch: null", "  micro_batch: 32"),
    ("configs/recipe.yaml", "  init_scale: null", "  init_scale: 65536.0"),
    ("configs/recipe.yaml", "  growth_interval: null", "  growth_interval: 2000"),
    ("configs/data.yaml", "  sha256: null ", "  sha256: " + "b" * 64 + " "),
    ("configs/data.yaml", "dataset_sha256: null ", "dataset_sha256: " + "a" * 64 + " "),
 ],
 "X3 commitments ladder l6_l8 (RUNBOOK 3.4)": [("configs/commitments.yaml", "\nladder: full", "\nladder: l6_l8")],
 "X4 SWA opt path flex (RUNBOOK 1.2)": [("configs/variants/swa256.yaml", "opt_path: chunked", "opt_path: flex")],
}
for label in sys.argv[1:]:
    key = next(k for k in CASES if k.startswith(label))
    files = set()
    for file, old, new in CASES[key]:
        p = WT / file; text = p.read_text()
        assert text.count(old) == 1, (file, old, text.count(old))
        p.write_text(text.replace(old, new)); files.add(file)
    try:
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-m", "not gpu and not t4"],
                              cwd=WT, env=env, capture_output=True, text=True, timeout=3000)
    finally:
        for file in files:
            subprocess.run(["git", "checkout", "--", file], cwd=WT, check=True)
    lines = proc.stdout.strip().splitlines()
    failed = [l for l in lines if l.startswith(("FAILED", "ERROR"))]
    print("==", key, "->", lines[-1])
    by_file = {}
    for l in failed:
        by_file.setdefault(l.split("::")[0].split()[-1], []).append(l)
    for f, ls in sorted(by_file.items()):
        print(f"   {f}: {len(ls)}  e.g. {ls[0][:230]}")
    (Path(__file__).parent / f"phil_{label}.log").write_text(proc.stdout[-200000:])
print("git status:", repr(subprocess.run(["git", "status", "--short"], cwd=WT, capture_output=True, text=True).stdout))

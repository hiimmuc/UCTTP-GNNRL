# UCTTP — GNN + RL for curriculum-based course timetabling

Graph neural network encoders and reinforcement learning agents for the curriculum-based
course timetabling problem (CB-CTT), evaluated against classical baselines on the 61 public
`.ectt` instances.

## Quick start

One shell script bootstraps the environment; everything else is a Python module.

```bash
scripts/setup.sh                 # virtualenv (Python 3.12) + dependencies + compiled validator
source .venv/bin/activate        # or prefix the commands below with `uv run --no-sync --`

python -m horarium.cli.doctor        # what torch can actually see; fails if a GPU is present but unusable
python -m horarium.cli.prepare       # fetch and verify the 61 instances, regenerate the derived docs
python -m horarium.cli.verify_cost   # BLOCKING: our cost function must equal the reference validator's
python -m horarium.cli.check         # ruff + mypy --strict + pytest  (add --fix to autoformat)
```

Train, then run the policy and check what it produced:

```bash
python -m horarium.cli.train --instance data/raw/Test/toy.ectt --formulation UD2 \
    --encoder flat --seed 0 --device cuda --checkpoint experiments/toy.pt
python -m horarium.cli.solve --instance data/raw/Test/toy.ectt --formulation UD2 \
    --checkpoint experiments/toy.pt --seed 0 --device cuda
```

`train` writes a checkpoint; `solve` reloads it, rolls the policy out, writes a `.sol`, and
cross-checks the cost against the reference validator, exiting non-zero if the two disagree or
if no attempt is feasible.

The encoder sweep and its report:

```bash
python -m horarium.cli.sweep --device cuda --seed 0                        # -> experiments/runs.jsonl
python -m horarium.cli.watch                                               # live progress bar
python -m horarium.cli.report                                              # -> docs/encoder_comparison.md + figures
```

## Project structure

```text
external/validator/   vendored validator.cc — ground truth for the UD1–UD5 cost functions
src/horarium/
  data/               .ectt and .sol readers/writers, corpus download, checksums
  problem/            instances, solutions, formulations, cost — pure semantics, no torch, no IO
  graph/              conflict graph and the feature blocks encoders consume
  envs/               Gymnasium environments
  models/             encoders, action head, masking
  agents/             PPO, checkpointing, inference rollouts
  eval/               validator bridge, corpus statistics
  cli/                entrypoints — `python -m horarium.cli.<command>`
data/raw/             instances, fetched by `python -m horarium.cli.prepare` (not committed; MANIFEST.json pins them)
```

`problem/` imports neither torch nor any IO module, enforced by `tests/test_purity.py`.

## Findings

- **`ROOM_CONSTRAINTS` lists rooms a course may *not* use.** `Instance.forbidden_rooms` holds
  them as written; `Instance.permitted_rooms` is the complement, materialised for masking.
- **The header counts constraint *rows*, and three instances repeat some.** `EA10`, `test2` and
  `test3` hold 32 duplicate rows between them. Repeats are idempotent, so this changes no cost.
- **UD4 makes room constraints hard**, not merely differently weighted, so `Formulation` carries
  a `room_constraints_are_hard` flag alongside the weights.
- **Some costs cannot be avoided.** Courses whose students outnumber their largest permitted room,
  and courses wanting more working days than they have lectures, put a floor under the optimum.
  `docs/instance_statistics.md` lists the floor per instance.

## Status

Phases 0–2 and 4–5 of `../docs/INSTRUCTION.md` are built, plus the train → checkpoint →
solve → validate loop and the encoder sweep. Phase 3 (MIP solvers) and Phases 6–9 are not.
See `docs/PHASE_REPORT.md` for what each gate produced.

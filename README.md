# kaggle-pipeline

Autonomous-but-gated Kaggle Playground pipeline. Claude does the thinking in 10-min chunks; GitHub Actions + Kaggle Notebooks do the heavy lifting. You approve each iteration via PR.

See [CLAUDE.md](CLAUDE.md) for the role contracts each Claude session follows.

## One-time setup

1. **Kaggle account** — join your target Playground comp (accept rules). Create an API token at <https://www.kaggle.com/settings> → "Create New Token". Save `kaggle.json`.
2. **GitHub repo** — create a new **private** repo (any name). Push this directory to it:
   ```bash
   cd kaggle-pipeline
   git remote add origin git@github.com:<you>/<repo>.git
   git push -u origin main
   ```
3. **Repo secrets** — Settings → Secrets and variables → Actions:
   - `KAGGLE_USERNAME` (from `kaggle.json`)
   - `KAGGLE_KEY` (from `kaggle.json`)
4. **Repo variables** — Settings → Secrets and variables → Actions → Variables tab:
   - `COMP_SLUG` — the competition slug, e.g. `playground-series-s6e4`
5. **Actions permissions** — Settings → Actions → General → Workflow permissions → **Read and write permissions** + **Allow GitHub Actions to create and approve pull requests**.
6. **Local bootstrap** (optional, for running `verify_setup.py` locally):
   ```bash
   bash scripts/bootstrap.sh
   COMP_SLUG=<slug> python scripts/verify_setup.py
   ```
7. **Pick a competition** — fill in `research/competition_brief.md` with the target, metric, data schema, and deadline.

## Running one iteration (happy path)

1. **Strategy PR** (Claude `strategist` role) → you merge to greenlight.
2. **Iteration PR** (Claude `coder` role) → you merge. This triggers `run-iteration.yml`, which pushes the kernel, waits, pulls output.
3. **Results PR** opens automatically with `kernel_output/`. Inspect `metrics.json`.
4. **Submit** — go to Actions → `submit` → "Run workflow", enter the iteration name. Public LB score is written to `state/leaderboard.json` via a follow-up PR.
5. **Review PR** (Claude `reviewer` role) closes the loop with `review.md` and the next ideas.

## Directory map

See the top of [the plan](../.claude/plans/i-am-struggling-to-fluttering-muffin.md) for the repo layout. Key entry points:

- `pipeline/src/train.py` — config-driven training orchestrator. Runs on Kaggle.
- `scripts/build_notebook.py` — assembles a pushable kernel directory from an iteration's `config.yaml`.
- `.github/workflows/run-iteration.yml` — pushes the kernel, polls, pulls output.
- `.github/workflows/submit.yml` — submits `submission.csv` and records public LB.
- `iterations/001_baseline/config.yaml` — the seed iteration (LGBM, 5-fold).

## Changing the competition

Update `vars.COMP_SLUG` in repo settings, wipe `state/leaderboard.json`, archive or delete `iterations/`, fill in a fresh `research/competition_brief.md`, drop a new `iterations/001_baseline/config.yaml`. The pipeline is otherwise comp-agnostic.

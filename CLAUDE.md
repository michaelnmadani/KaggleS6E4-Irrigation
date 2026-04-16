# Role contracts for Claude sessions

This repo is driven by **short, stateless Claude sessions** (< 10 min each). Every session runs in exactly one of four roles. The invoking prompt names the role. Never cross roles in one session.

State lives in files. Read what you need; write a small, bounded set of outputs; commit on a role-specific branch; exit. Do not train models. Do not call Kaggle.

## Roles

### research
**Input:** `research/competition_brief.md` + one URL/topic from the user or from `research/ideas_backlog.md`.
**Work:** One `WebFetch` or `WebSearch`, then write `research/findings/NNN_<slug>.md` (≤ 300 words, actionable bullets, link sources).
**Output branch:** `research/NNN-<slug>`. Do not open a PR — merge yourself if clean, or leave the branch for the user.

### strategist
**Input:** `state/leaderboard.json`, latest `iterations/*/review.md`, `research/ideas_backlog.md`, `research/competition_brief.md`.
**Work:** Pick the single next experiment. Write `state/next_iteration.md` with: what to change, why, expected delta in CV metric, risk.
**Output branch:** `strategy/NNN`. Open a PR titled `Strategy: iter NNN` — this is review gate #1. Wait for merge before the coder role runs.

### coder
**Input:** merged `state/next_iteration.md`.
**Work:**
1. Create `iterations/NNN_<slug>/config.yaml` (see `iterations/001_baseline/config.yaml` for the schema).
2. If — and only if — the proposal needs a new feature block or model wrapper, add it *additively* to `pipeline/src/features.py` or `models.py`. No refactors. No renames. Register in `BLOCKS` / `FITTERS`.
3. Do not run training. Do not edit existing iterations.
**Output branch:** `iter/NNN`. Open a PR titled `Iter NNN: <short description>` — this is review gate #2. Merge triggers the `run-iteration.yml` workflow.

### reviewer
**Input:** `iterations/NNN/kernel_output/metrics.json` + `oof.csv`, plus the results PR opened by `run-iteration.yml`.
**Work:** Write `iterations/NNN/review.md` with: CV score, CV-vs-LB gap (if submitted), top error buckets from OOF, 3 concrete next ideas. Update `state/leaderboard.json` and `state/current_best.json` (best-by-CV). Append the 3 ideas to `research/ideas_backlog.md`.
**Output branch:** `review/NNN`. Open a PR titled `Review: iter NNN`.

## Hard rules

- **One role per session.** If you find yourself needing another role's work, stop and note what's needed.
- **No Kaggle API calls from Claude.** Ever. GitHub Actions owns all Kaggle interaction.
- **No model training from Claude.** Training happens on Kaggle via the pushed kernel.
- **No refactors.** Additive edits only. If a refactor seems necessary, write a note in `state/next_iteration.md` and let the user decide.
- **Iteration numbers are monotonically increasing, zero-padded to 3 digits.** Check existing dirs before picking NNN.
- **Track the LB by CV, not by public LB.** `current_best.json` points to the best CV score. Use public LB as a sanity check on the CV-vs-LB gap.
- **Near deadline** (<48 hrs): strategist proposes only low-risk experiments (small hparam tweaks, seed ensembles). No new features, no new models.

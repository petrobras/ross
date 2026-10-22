---
name: ross-pr-review
description: >-
  Review a pull request to the ROSS repository (ross-rotordynamics) against
  AGENTS.md: scope, architecture (elements, Rotor.run_* methods, Results),
  test integrity, writing, PR text and correctness, with an optional HTML
  report. Use when asked to review a ROSS pull request by number, URL or
  branch. The review goes to the person who asked; nothing is posted to
  GitHub until they approve it.
argument-hint: "[PR number or branch]"
license: Apache-2.0
---

# Review a ROSS pull request

## Rules

- **Nothing reaches GitHub without approval.** Do not comment, review,
  approve, request changes, label or push until the person who asked for the
  review has read the exact text and approved posting it.
- **Independence.** Judge the PR from its text, its diff, the facts and your
  own runs, not from the author's explanations. If this session wrote or
  edited the code under review, say so at the top of the review and suggest
  repeating it in a new session, ideally with a different model.
- **Static before dynamic.** Collect the facts with the `tools/pr_facts.py`
  of the checkout you are working in, never the PR's copy, so that a PR
  cannot change its own checks. Read the changes to tests, `conftest.py`,
  CI and packaging before running any of the PR's code. The PR's code runs on
  this machine: if it does anything unrelated to the change (network access,
  writing outside the repository, reading credentials), stop and report it.
- **Findings that matter.** Report what affects correctness, scope,
  architecture, test integrity or readability, each with `file:line`.
  A correctness finding needs a reproducer that you ran; without one it is a
  question. Rank them by their effect on users, keep the important ones
  (about five at most), and make no style comments beyond the `AGENTS.md`
  rules.
- Decisions that belong to the maintainer (scope, naming, defaults,
  deprecation aliases) are questions for them, not verdicts.

## 1. Set up

Run everything from the root of a ROSS checkout. Use the Python of the
environment ROSS's dependencies are installed in (for example
`.venv/bin/python`); a bare `python` may be a different interpreter.

```bash
N=<number>
OUT="$(git rev-parse --git-common-dir)/ross-review/pr-$N"
mkdir -p "$OUT/probes"
git fetch upstream main "+pull/$N/head:refs/review/pr-$N"
git rev-parse --short refs/review/pr-$N     # confirms the PR head was fetched
cat > "$OUT/env.sh" <<EOF
N=$N
OUT="$OUT"
PY=<path to the environment's python>
RUFF=<path to the environment's ruff>
HEAD_REF=refs/review/pr-$N
BASE=$(git merge-base upstream/main refs/review/pr-$N)
EOF
gh pr view $N -R petrobras/ross --json number,title,author,body,headRefOid,isDraft,url > "$OUT/pr.json"
```

Shell variables do not survive between separate tool calls; start each one
with `source "$OUT/env.sh"` (with `OUT` spelled out). For a local branch
instead of a PR, write the branch into `HEAD_REF` and skip `gh`. `OUT` is
inside the main repository's `.git` (shared by all its worktrees), so
nothing from the review is committed and every review is found in one place.

## 2. Collect the facts

```bash
$PY tools/pr_facts.py --base $BASE --head $HEAD_REF > "$OUT/facts.md"
```

Read all of it. The flags are prompts: confirm or dismiss each one by looking
at the code. Pay special attention to edited fixtures and helpers in test
files: a changed fixture can keep old reference values passing after a
behavior change.

## 3. Read the PR text

Check the sections of the PR template as it exists on the PR's base
(`git show $BASE:.github/PULL_REQUEST_TEMPLATE.md 2>/dev/null || echo "no
template on the base"`); without one, check that the text covers scope,
design decisions, API changes, evidence, changes to existing tests and
documentation. Compare the claims with the facts: "API changes: none" while
the facts list changes, "Changes to existing tests: none" while there are
`tests` flags, evidence that names no reference. Note every mismatch, and
reproduce the key numbers the text claims (step 6).

## 4. Decide how deep to go

| Tier | When | Depth |
|---|---|---|
| T0 mechanical | docs prose, notebook outputs, CI pins, formatting, vendored files alone | facts and PR text |
| T1 peripheral | interface, `ross_2to3`, examples, recipes, tools, test-only changes, fixes with no API change | plus reading the diff |
| T2 API | anything in the facts' "Public API" section | plus running tests and probes; HTML report when the change has physics or numbers worth showing |
| T3 constitution | scope flags; `constitution` flags; changes to the global matrices or the save/load format; reference data | plus one critical path traced end to end; HTML report |

Use the highest tier that applies, and state it in the review. A
`constitution` flag marked "whitespace only" does not raise the tier. For T3,
record the traced path in the card's Evidence column as a chain of `file:line`
hops (for example, how a new element's `K` reaches `Rotor.K`).

## 5. The checks

- **Scope.** Compare with the Scope section of `AGENTS.md`. Verdict: core,
  belongs elsewhere, or maintainer decides, quoting the line it rests on.
- **Architecture.** Go through the `architecture` and `api` flags and read
  each new class. Check that the change reuses what exists: search for
  similar functionality (`git grep -n "<concept>" $BASE -- ross`). Check that
  names follow the existing ones (`run_*`, `plot_*`, `*_example`, `from_*`).
- **Test integrity.** For every `tests` flag, read the diff of that test
  or fixture (`git diff $BASE $HEAD_REF -- <file>`) and decide whether the
  change is justified, and justified in the PR text. A widened tolerance is
  justified when it follows the tolerance rule in AGENTS.md *Tests*:
  a named source, a size derived from it rather than from the failing value,
  a margin below the effect the test checks, and one tolerance on every
  platform. It is a finding when the change also moves results. For new physics, find
  where each expected value comes from: literature, an analytical case, an
  independent path, or (a finding) the code under test. Check that the tests
  exercise what changed (`MultiRotor`, `save`/`load`, units)
  over a range, not a single point.
- **Writing.** Read the new public docstrings and the changed documentation.
  Quote passages that repeat, restate the signature, narrate the process or
  pad; for long ones, say whether the length carries information. Notebooks
  flagged as stale must be re-executed.
- **PR text.** Complete and accurate, from step 3.

## 6. Correctness

Check out the PR, and the base for comparisons, in worktrees:

```bash
git worktree add "$OUT/tree" $HEAD_REF
git worktree add "$OUT/base-tree" $BASE
(cd "$OUT/tree" && $RUFF check ross && $RUFF format --check ross)
(cd "$OUT/tree" && PYTHONPATH="$PWD" $PY -m pytest -p no:cacheprovider -n auto \
    <changed and new tests> <changed modules>)
```

If the PR adds a dependency that is not installed, say so instead of
installing it into the shared environment.

Then probe what the tests do not cover. Write each probe as a script in
`"$OUT/probes/"`, run it against both trees (`PYTHONPATH="$OUT/tree"` and
`PYTHONPATH="$OUT/base-tree"`) where the comparison matters, and save its
output next to it (`p1_name.py` → `p1_name.out`):

- the new element or option inside `Rotor` and `MultiRotor`;
- behavior that should not change: the same model on base and head, with
  the new options left at their defaults;
- `save`/`load` round trips of new elements and results;
- units: pass `Q_` quantities where `@check_units` applies;
- limiting cases with known answers (zero speed, zero damping, rigid
  supports, an analytical reduced model) and a comparison with an
  independent ROSS path when one exists;
- a sweep beyond the parameter points the tests use;
- the same model built another way (directly instead of through a new
  helper, or with an independent mesh) to tell the PR's effect from existing
  behavior;
- the numbers the PR text claims.

For each correctness finding, trace its consumers (`git grep -n` for the
attribute or function) and say what a user would see: wrong numbers, a
failing call, or only internal state.

Analysis results are cached per rotor instance (`run_modal` uses
`lru_cache`): build a fresh rotor for each parameter value in a probe.
Running the PR's new tests on the base rarely works (shared fixtures use the
new API), so check "fails without the fix" with a probe instead.

Keep the worktrees until the HTML report is built (step 8); then remove them
with `git worktree remove --force "$OUT/tree"` and the same for
`"$OUT/base-tree"`.

## 7. Write the review card

Fill in [review-card.md](review-card.md), save it as `"$OUT/review.md"`, and
show it in the conversation.

## 8. HTML report

For T3, for T2 when step 4 calls for it, or when asked, build an HTML report with
[scripts/report.py](scripts/report.py). Write the builder as
`"$OUT/build_report.py"` and run it from the checkout root with
`PYTHONPATH="$OUT/tree" $PY "$OUT/build_report.py"`:

```python
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, ".agents/skills/ross-pr-review/scripts")
from report import Report

import ross

N = <number>
assert "ross-review" in ross.__file__, "run with PYTHONPATH=$OUT/tree"
git_dir = subprocess.check_output(["git", "rev-parse", "--git-common-dir"], text=True)
OUT = Path(git_dir.strip()) / f"ross-review/pr-{N}"
HEAD = f"refs/review/pr-{N}"
BASE = subprocess.check_output(["git", "merge-base", "upstream/main", HEAD], text=True)

report = Report(title=f"#{N} <PR title>", head=HEAD, base=BASE.strip())
report.markdown((OUT / "review.md").read_text())
report.section("Theory", "<p>...</p>")
report.code("ross/<module>.py", symbol="<Class.method>")
report.heading("Results")
report.figure(fig, "<what the figure compares>")
report.save(OUT / "report.html")
```

Order: the review card first, then theory, the implementation gist (code
excerpts from `report.code`, read from the PR's head and never retyped),
usage examples and results. Every number and figure comes from code the
builder runs. The report stays local unless the user asks to share it. To
rebuild it after the PR changes, fetch the PR again, recreate the worktrees
and rerun the builder.

## 9. Posting, only after approval

Ask how to post: a comment, "request changes" or "approve", and whether to
include inline comments. Show the exact text, then, for example:

```bash
gh pr review $N -R petrobras/ross --comment --body-file "$OUT/review.md"
```

Use `--request-changes` or `--approve` only when the user chose it. Once the
review is posted or dropped, `git update-ref -d refs/review/pr-$N` removes the
fetched head.

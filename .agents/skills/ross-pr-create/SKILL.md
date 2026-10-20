---
name: ross-pr-create
description: >-
  Prepare a change to the ROSS repository (ross-rotordynamics) for a pull
  request: check it against AGENTS.md (scope, architecture, test integrity,
  writing), run the checks, and write the PR text from the template. Use when
  asked to prepare, finish, describe or open a pull request in the ROSS
  repository, or before pushing a branch for review.
license: Apache-2.0
---

# Prepare a ROSS pull request

`AGENTS.md` (section "Developing ROSS") holds the rules; this skill is the
procedure for meeting them before a PR is opened. The goal is a PR that a
maintainer can judge from its text and from the output of `tools/pr_facts.py`,
without reconstructing what was done.

Work from the repository root, on the feature branch, with the Python of the
environment ROSS's dependencies are installed in (for example
`.venv/bin/python`; a bare `python` may be a different interpreter).

## 1. Collect the facts

```bash
git fetch upstream
python tools/pr_facts.py          # working tree vs merge base with upstream/main
```

Without an `upstream` remote, pass `--base origin/main`. Read the whole output.
It lists flags (size, scope, architecture, api, tests, writing), the files by
category, the public API changes, the test findings and the writing signals.
Rerun it after each round of fixes. By the end, every flag is either fixed in
the code or explained in the PR text.

## 2. Scope

- A new subpackage, a new runtime dependency or a new physical domain must fit
  the **Scope** section of `AGENTS.md`. If it is core, note why for the PR's
  Scope section. If it belongs elsewhere, or under "Not decided yet, ask
  first", or you cannot tell: **stop and ask the user** before doing more work.
- Size flags (core and interface mixed, vendored files mixed with other
  changes, more than about 1,000 hand-written lines): propose a split to the
  user as an ordered list of PRs that each stand alone, with the tests next to
  the code they test. Split only after the user agrees.

## 3. Architecture and API

Resolve each `architecture` and `api` flag:

| Flag | Fix |
|---|---|
| `run_*` method on a class that is not a rotor | Move the analysis to `Rotor` (override in `MultiRotor` when they need it), returning a `Results` subclass |
| Element does not define `M`, `K`, `C`, `G`, `dof_mapping` | Implement them; add tests that build the element in `Rotor` and `MultiRotor` and round-trip `save`/`load` |
| New rotor `run_*` missing from `AGENTS.md` or `SKILL.md` | Add the row to the table in `AGENTS.md`, the recipe in `ross/agent_skills/ross/` and the row in its `SKILL.md` |
| Class named `*Results` that does not subclass `Results` | Subclass `Results`; add a `save`/`load` round-trip test |
| Type annotations | Move the type information to the docstring |
| Removed, renamed or moved public name; changed default; new parameter | List every item in the PR's API changes with its reason. A default that changes behavior for existing models is a decision for the maintainer: say so explicitly |

Then look for duplication the facts cannot see: search the code base for
existing classes, helpers or results types that already do what the change
adds (`git grep -n "<concept>" -- ross`). Reuse them, or say in the PR why
they do not fit.

## 4. Tests

- **Integrity flags** (`tests` group): a loosened tolerance, fewer
  assertions, an added `skip`/`xfail`, changed expected values, an edited
  fixture or helper, changed test configuration or reference data. An edited
  fixture deserves extra care: adding an argument to keep old reference values
  passing hides a change of default behavior. Undo each one unless it is physically
  or numerically justified (for a tolerance, see the tolerance rule
  in AGENTS.md *Tests*); if it is, keep it and write the reason in the PR's "Changes to
  existing tests or reference data". For a moved test, confirm that every
  assertion came along.
- **New physics** needs tests whose expected values do not come from the new
  code: literature (cite it next to the test), an analytical or limiting
  case, or a different ROSS code path. Cover a parameter range, not a single
  point.
- **A bug fix** needs a test that fails without the fix. Check it against the
  merge base:

  ```bash
  BASE=$(git merge-base upstream/main HEAD)
  git worktree add ../ross-pr-base "$BASE"
  cp ross/tests/test_<module>.py ../ross-pr-base/ross/tests/
  PYTHONPATH=../ross-pr-base python -m pytest ../ross-pr-base/ross/tests/test_<module>.py::<test>
  git worktree remove --force ../ross-pr-base
  ```

  The test should fail there and pass on the branch.
- Library code must not refer to pytest or silence exceptions to get a green
  run (the facts flag both).

## 5. Writing

- Every new public class, function and method has a NumPy docstring: an
  imperative summary line, parameters with meaning and unit, returns, a short
  doctest example where it helps, references for theory.
- For each long docstring flagged, keep what a user needs and cut
  repetition, restated signatures, narration of how the change was made, and
  filler. Length itself is fine when the content needs it.
- Filler phrases are flagged without context; rewrite them where they are
  filler and keep them where they are technical ("robust control").
- Text added several times (a citation pasted into three docstrings, the
  same advice repeated): keep it in one place, usually the docs page or
  `docs/references.bib`, and point to it.
- Notebooks flagged as stale: re-execute them so the stored outputs match
  the code.
- Documentation pages: extend an existing tutorial before adding a page;
  derivations go in the docs with references in `docs/references.bib`.

## 6. Run the checks

```bash
ruff check ross && ruff format ross
python -m pytest <changed test files> <changed modules, for their doctests>
python -m pytest ross -n auto      # full suite, before opening the PR
```

Record the counts (passed, skipped, failed). If a failure also happens on
`main`, name it in the PR rather than hiding it.

## 7. Write the PR text

Fill in every section of `.github/PULL_REQUEST_TEMPLATE.md`, writing "none"
where nothing applies, and save it outside the tracked tree:
`"$(git rev-parse --git-dir)/PR_BODY.md"`.

- **Summary**: what changes and why, in a few sentences.
- **Scope**: from step 2.
- **Design decisions for review**: the choices a maintainer should confirm,
  each with the alternatives rejected. Condense the local plan into this
  section; do not paste the plan, session notes or transcripts.
- **API changes**: every item in the "Public API" section of the facts, with
  its reason.
- **Evidence**: the test counts from step 6; for new physics, each reference
  case (source, quantity, tolerance, parameter range); figures or tables when
  they help.
- **Changes to existing tests or reference data**: every remaining `tests`
  flag, with its reason, or "none".
- **Documentation**: what was added, changed or removed.
- **Where to look**: the parts whose correctness you are least sure of. Be
  specific and honest; this is where a reviewer spends their time.
- Leave the CLA checkbox for the user to tick.

Show the PR text to the user.

## 8. Open the PR only when asked

Pushing and opening the PR are the user's decisions. When asked:

```bash
git push origin <branch>
gh pr create -R petrobras/ross --base main --head <fork-owner>:<branch> \
  --title "<imperative sentence>" --body-file "$(git rev-parse --git-dir)/PR_BODY.md"
```

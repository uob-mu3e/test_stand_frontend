# `git`

## `main` / `dev` branches and release strategy

- `main` branch always points to latest release tag
- the development branch is `dev`
- at the release point the `dev` branch is merged into `main` branch
  and tagged as `release/${yy}.${mm}` or `release/${yy}.${mm}p{n}`,
  where `${yy}` is 2 digit year and `${mm}` is 2 digit month,
  and `p${n}` is optional patch number (for bug fixes)

- merged or not needed branches are tagged and deleted,
  for branch `${branch_name}` the is `archive/${yyyyMMdd}_${branch_name}`,
  where `${yyyyMMdd}` is the date of last commit on this branch

## pull requests (PR)

- the development should be done on separate branches
- the PR should be submitted against `dev` branch
- there should be at least 2 reviewer for PR:
  maintainer of repository
  and person responsible (or expert) of the subsystem
- at least one approval is required to merge the PR

## house rules

- try to keep up to date with `dev` branch
- do not commit binary files, especially ones that can be generated
- respect the project structure
- try to reuse existing code
- do not commit all your files blindly (check the diff before committing)
- avoid self merges, i.e. "merge 'origin/branch' into 'branch'",
  instead use `git rebase`

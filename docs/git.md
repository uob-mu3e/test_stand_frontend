# git FAQ

## `master` / `dev` branches and release strategy

- the development branch is `dev`
- at the release point the `dev` branch is merged into `master` branch
  and tagged as `${yy}.${mm}` or `${yy}.${mm}p{n}`,
  where `${yy}` is 2 digit year and `${mm}` is 2 digit month,
  and `p${n}` is optional patch number (for bug fixes)
- `master` branch always points to latest release tag

- merged or not needed branches are tagged and deleted,
  for branch `${branch_name}` the is `archive/${yyyyMMdd}_${branch_name}`,
  where `${yyyyMMdd}` is the date of last commit on this branch

## house rules

- do development on separate branches
- if you want to submit a fix,
  make a new branch with your changes and then make pull request (PR)
- the pull requests should into the `dev` branch
- try to keep up to date with `dev` branch

- do not commit binary files, especially ones that can be generated
- respect the project structure
- try to reuse existing code
- do not commit all your files blindly (check the diff before committing)

- avoid self merges, i.e. "merge 'origin/branch' into 'branch'",
  instead use `git rebase`

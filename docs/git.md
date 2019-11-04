# git FAQ

- do development on _your_ branch
- if you want to submit fix, make new branch with your changes and then make pull request
- make pull request to latest dev branch
- merge dev branch to your branch as often as possible

- do not commit binary files, especially ones that can be generated
- respect the project structure
- try to reuse existing code
- do not commit blindly all files (check you diff before committing)

- if you want to close branch `${branch_name}`,
  tag it as `archive/${yyyyMMdd}_${branch_name}`,
  where `${yyyyMMdd}` is the date of last commit on this branch

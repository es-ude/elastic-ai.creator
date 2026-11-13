Please have a look at our [contribution guidelines](https://es-ude.github.io/elastic-ai.creator/contribution.html) before submitting a pull request.

## Description

<!-- What changes did you make and why? -->

## Type of change

<!-- Keep the conventional commit types that apply. -->

- [ ] `feat`: new feature
- [ ] `fix`: bug fix
- [ ] `docs`: documentation change
- [ ] `style`: code style/formatting
- [ ] `refactor`: code refactoring
- [ ] `perf`: performance improvement
- [ ] `test`: testing changes
- [ ] `chore`: build/tooling changes
- [ ] `ci`: CI configuration changes
- [ ] `BREAKING CHANGE`

## Breaking changes

<!-- List any breaking changes and explain why they are necessary. Remove this section if there are none. -->

## Checklist

- [ ] I have added/updated tests that prove my fix/feature works
- [ ] All new and existing tests pass
- [ ] I have formatted my code using the project's style guidelines
- [ ] I have rebased my branch on the latest main
- [ ] My commits follow a linear history (no merge commits)
- [ ] My commit messages are descriptive and explain what/why (e.g., "feat(component): add validation to reduce server load")

## Development guidelines

- Maintain a linear history by rebasing your branch before merging
- Format your code according to project guidelines
- Write meaningful tests for new features and bug fixes
- Breaking changes must be clearly documented
- Each commit should be atomic and have a descriptive message explaining what change was made and why it was necessary

### Code quality tools

- **Ruff** ([docs](https://docs.astral.sh/ruff/)): `ruff check` for linting, `ruff format` for formatting
- **MyPy** ([docs](https://mypy.readthedocs.io)): `mypy -p elasticai.creator` for type checking
- **Cocogitto** ([docs](https://docs.cocogitto.io)): `cog` for commit message linting
- **Pre-commit**: use our `.pre-commit-config.yaml` to run some checks automatically:

  ```bash
  pip install pre-commit
  pre-commit install
  ```

### Version control tips

Consider using [Jujutsu](https://github.com/martinvonz/jj) (`jj`) for easier maintenance of a linear history:

```bash
jj new main  # create new change on top of main
jj squash    # combine current change into previous one
jj rebase -r <id1> -A <id2>   # move change <id1> behind <id2> in the history
jj describe  # edit the commit message for the current change
jj git push -c <id>  # push the given change id to a new branch on github
```

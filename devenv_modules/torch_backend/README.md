# Torch Backend Devenv Module

The module only makes sense together with this projects `pyproject.toml`.
It allows us to inject the index for the pytorch backend from command line
or via devenv.local.nix.

E.g., inside CI we can force to download the cpu backend by running

```bash
devenv --option torch.backend:string "cpu" shell
```

in the shell.

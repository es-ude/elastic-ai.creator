# {py:mod}`elasticai.creator.vhdl.skeleton_id`

```{py:module} elasticai.creator.vhdl.skeleton_id
```

```{autodoc2-docstring} elasticai.creator.vhdl.skeleton_id
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`compute_skeleton_id_hash <elasticai.creator.vhdl.skeleton_id.compute_skeleton_id_hash>`
  - ```{autodoc2-docstring} elasticai.creator.vhdl.skeleton_id.compute_skeleton_id_hash
    :summary:
    ```
* - {py:obj}`replace_id_in_vhdl <elasticai.creator.vhdl.skeleton_id.replace_id_in_vhdl>`
  - ```{autodoc2-docstring} elasticai.creator.vhdl.skeleton_id.replace_id_in_vhdl
    :summary:
    ```
* - {py:obj}`update_skeleton_id_in_build_dir <elasticai.creator.vhdl.skeleton_id.update_skeleton_id_in_build_dir>`
  - ```{autodoc2-docstring} elasticai.creator.vhdl.skeleton_id.update_skeleton_id_in_build_dir
    :summary:
    ```
````

### API

````{py:function} compute_skeleton_id_hash(files: collections.abc.Iterable[pathlib.Path]) -> bytes
:canonical: elasticai.creator.vhdl.skeleton_id.compute_skeleton_id_hash

```{autodoc2-docstring} elasticai.creator.vhdl.skeleton_id.compute_skeleton_id_hash
```
````

````{py:function} replace_id_in_vhdl(code: collections.abc.Iterable[str], id: bytes) -> collections.abc.Iterable[str]
:canonical: elasticai.creator.vhdl.skeleton_id.replace_id_in_vhdl

```{autodoc2-docstring} elasticai.creator.vhdl.skeleton_id.replace_id_in_vhdl
```
````

````{py:function} update_skeleton_id_in_build_dir(build_dir: pathlib.Path) -> None
:canonical: elasticai.creator.vhdl.skeleton_id.update_skeleton_id_in_build_dir

```{autodoc2-docstring} elasticai.creator.vhdl.skeleton_id.update_skeleton_id_in_build_dir
```
````

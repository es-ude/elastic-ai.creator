# {py:mod}`elasticai.creator.file_generation.on_disk_path`

```{py:module} elasticai.creator.file_generation.on_disk_path
```

```{autodoc2-docstring} elasticai.creator.file_generation.on_disk_path
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`OnDiskFile <elasticai.creator.file_generation.on_disk_path.OnDiskFile>`
  -
* - {py:obj}`OnDiskPath <elasticai.creator.file_generation.on_disk_path.OnDiskPath>`
  -
````

### API

`````{py:class} OnDiskFile(full_path: str)
:canonical: elasticai.creator.file_generation.on_disk_path.OnDiskFile

Bases: {py:obj}`elasticai.creator.file_generation.savable.File`

````{py:method} write(template: elasticai.creator.file_generation.template.Template) -> None
:canonical: elasticai.creator.file_generation.on_disk_path.OnDiskFile.write

```{autodoc2-docstring} elasticai.creator.file_generation.on_disk_path.OnDiskFile.write
```

````

`````

`````{py:class} OnDiskPath(name: str, parent: str = '.')
:canonical: elasticai.creator.file_generation.on_disk_path.OnDiskPath

Bases: {py:obj}`elasticai.creator.file_generation.savable.Path`

````{py:method} create_subpath(name: str) -> elasticai.creator.file_generation.on_disk_path.OnDiskPath
:canonical: elasticai.creator.file_generation.on_disk_path.OnDiskPath.create_subpath

```{autodoc2-docstring} elasticai.creator.file_generation.on_disk_path.OnDiskPath.create_subpath
```

````

````{py:method} as_file(suffix: str) -> elasticai.creator.file_generation.on_disk_path.OnDiskFile
:canonical: elasticai.creator.file_generation.on_disk_path.OnDiskPath.as_file

```{autodoc2-docstring} elasticai.creator.file_generation.on_disk_path.OnDiskPath.as_file
```

````

`````

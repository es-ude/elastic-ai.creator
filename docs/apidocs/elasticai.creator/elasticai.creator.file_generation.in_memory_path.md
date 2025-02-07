# {py:mod}`elasticai.creator.file_generation.in_memory_path`

```{py:module} elasticai.creator.file_generation.in_memory_path
```

```{autodoc2-docstring} elasticai.creator.file_generation.in_memory_path
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`InMemoryFile <elasticai.creator.file_generation.in_memory_path.InMemoryFile>`
  -
* - {py:obj}`InMemoryPath <elasticai.creator.file_generation.in_memory_path.InMemoryPath>`
  -
````

### API

`````{py:class} InMemoryFile(name: str)
:canonical: elasticai.creator.file_generation.in_memory_path.InMemoryFile

Bases: {py:obj}`elasticai.creator.file_generation.savable.File`

````{py:method} write(template: elasticai.creator.file_generation.template.Template) -> None
:canonical: elasticai.creator.file_generation.in_memory_path.InMemoryFile.write

```{autodoc2-docstring} elasticai.creator.file_generation.in_memory_path.InMemoryFile.write
```

````

`````

`````{py:class} InMemoryPath(name: str, parent: typing.Optional[elasticai.creator.file_generation.in_memory_path.InMemoryPath])
:canonical: elasticai.creator.file_generation.in_memory_path.InMemoryPath

Bases: {py:obj}`elasticai.creator.file_generation.savable.Path`

````{py:method} as_file(suffix: str) -> elasticai.creator.file_generation.in_memory_path.InMemoryFile
:canonical: elasticai.creator.file_generation.in_memory_path.InMemoryPath.as_file

```{autodoc2-docstring} elasticai.creator.file_generation.in_memory_path.InMemoryPath.as_file
```

````

````{py:method} __getitem__(item: str) -> InMemoryPath | InMemoryFile
:canonical: elasticai.creator.file_generation.in_memory_path.InMemoryPath.__getitem__

```{autodoc2-docstring} elasticai.creator.file_generation.in_memory_path.InMemoryPath.__getitem__
```

````

````{py:method} create_subpath(subpath_name: str) -> elasticai.creator.file_generation.in_memory_path.InMemoryPath
:canonical: elasticai.creator.file_generation.in_memory_path.InMemoryPath.create_subpath

```{autodoc2-docstring} elasticai.creator.file_generation.in_memory_path.InMemoryPath.create_subpath
```

````

`````

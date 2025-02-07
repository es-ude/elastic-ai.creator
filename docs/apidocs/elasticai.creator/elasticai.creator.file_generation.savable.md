# {py:mod}`elasticai.creator.file_generation.savable`

```{py:module} elasticai.creator.file_generation.savable
```

```{autodoc2-docstring} elasticai.creator.file_generation.savable
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`File <elasticai.creator.file_generation.savable.File>`
  -
* - {py:obj}`Path <elasticai.creator.file_generation.savable.Path>`
  -
* - {py:obj}`Savable <elasticai.creator.file_generation.savable.Savable>`
  -
````

### API

`````{py:class} File
:canonical: elasticai.creator.file_generation.savable.File

Bases: {py:obj}`typing.Protocol`

````{py:method} write(template: elasticai.creator.file_generation.template.Template) -> None
:canonical: elasticai.creator.file_generation.savable.File.write
:abstractmethod:

```{autodoc2-docstring} elasticai.creator.file_generation.savable.File.write
```

````

`````

`````{py:class} Path
:canonical: elasticai.creator.file_generation.savable.Path

Bases: {py:obj}`typing.Protocol`

````{py:method} as_file(suffix: str) -> elasticai.creator.file_generation.savable.File
:canonical: elasticai.creator.file_generation.savable.Path.as_file
:abstractmethod:

```{autodoc2-docstring} elasticai.creator.file_generation.savable.Path.as_file
```

````

````{py:method} create_subpath(subpath_name: str) -> elasticai.creator.file_generation.savable.Path
:canonical: elasticai.creator.file_generation.savable.Path.create_subpath
:abstractmethod:

```{autodoc2-docstring} elasticai.creator.file_generation.savable.Path.create_subpath
```

````

`````

`````{py:class} Savable
:canonical: elasticai.creator.file_generation.savable.Savable

Bases: {py:obj}`typing.Protocol`

````{py:method} save_to(destination: elasticai.creator.file_generation.savable.Path) -> None
:canonical: elasticai.creator.file_generation.savable.Savable.save_to
:abstractmethod:

```{autodoc2-docstring} elasticai.creator.file_generation.savable.Savable.save_to
```

````

`````

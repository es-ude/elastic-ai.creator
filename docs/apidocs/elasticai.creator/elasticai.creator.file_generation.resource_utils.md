# {py:mod}`elasticai.creator.file_generation.resource_utils`

```{py:module} elasticai.creator.file_generation.resource_utils
```

```{autodoc2-docstring} elasticai.creator.file_generation.resource_utils
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_file_from_package <elasticai.creator.file_generation.resource_utils.get_file_from_package>`
  - ```{autodoc2-docstring} elasticai.creator.file_generation.resource_utils.get_file_from_package
    :summary:
    ```
* - {py:obj}`read_text <elasticai.creator.file_generation.resource_utils.read_text>`
  - ```{autodoc2-docstring} elasticai.creator.file_generation.resource_utils.read_text
    :summary:
    ```
* - {py:obj}`copy_file <elasticai.creator.file_generation.resource_utils.copy_file>`
  - ```{autodoc2-docstring} elasticai.creator.file_generation.resource_utils.copy_file
    :summary:
    ```
* - {py:obj}`get_full_path <elasticai.creator.file_generation.resource_utils.get_full_path>`
  - ```{autodoc2-docstring} elasticai.creator.file_generation.resource_utils.get_full_path
    :summary:
    ```
* - {py:obj}`read_text_from_path <elasticai.creator.file_generation.resource_utils.read_text_from_path>`
  - ```{autodoc2-docstring} elasticai.creator.file_generation.resource_utils.read_text_from_path
    :summary:
    ```
* - {py:obj}`save_text_to_path <elasticai.creator.file_generation.resource_utils.save_text_to_path>`
  - ```{autodoc2-docstring} elasticai.creator.file_generation.resource_utils.save_text_to_path
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PathType <elasticai.creator.file_generation.resource_utils.PathType>`
  - ```{autodoc2-docstring} elasticai.creator.file_generation.resource_utils.PathType
    :summary:
    ```
* - {py:obj}`Package <elasticai.creator.file_generation.resource_utils.Package>`
  - ```{autodoc2-docstring} elasticai.creator.file_generation.resource_utils.Package
    :summary:
    ```
````

### API

````{py:data} PathType
:canonical: elasticai.creator.file_generation.resource_utils.PathType
:value: >
   None

```{autodoc2-docstring} elasticai.creator.file_generation.resource_utils.PathType
```

````

````{py:data} Package
:canonical: elasticai.creator.file_generation.resource_utils.Package
:value: >
   None

```{autodoc2-docstring} elasticai.creator.file_generation.resource_utils.Package
```

````

````{py:function} get_file_from_package(package: elasticai.creator.file_generation.resource_utils.Package, file_name: str) -> typing.ContextManager[pathlib.Path]
:canonical: elasticai.creator.file_generation.resource_utils.get_file_from_package

```{autodoc2-docstring} elasticai.creator.file_generation.resource_utils.get_file_from_package
```
````

````{py:function} read_text(package: elasticai.creator.file_generation.resource_utils.Package, file_name: str) -> collections.abc.Iterator[str]
:canonical: elasticai.creator.file_generation.resource_utils.read_text

```{autodoc2-docstring} elasticai.creator.file_generation.resource_utils.read_text
```
````

````{py:function} copy_file(package: elasticai.creator.file_generation.resource_utils.Package, file_name: str, destination: elasticai.creator.file_generation.resource_utils.PathType) -> None
:canonical: elasticai.creator.file_generation.resource_utils.copy_file

```{autodoc2-docstring} elasticai.creator.file_generation.resource_utils.copy_file
```
````

````{py:function} get_full_path(package: elasticai.creator.file_generation.resource_utils.Package, file_name: str) -> str
:canonical: elasticai.creator.file_generation.resource_utils.get_full_path

```{autodoc2-docstring} elasticai.creator.file_generation.resource_utils.get_full_path
```
````

````{py:function} read_text_from_path(path: elasticai.creator.file_generation.resource_utils.PathType, encoding: str = 'utf-8') -> str
:canonical: elasticai.creator.file_generation.resource_utils.read_text_from_path

```{autodoc2-docstring} elasticai.creator.file_generation.resource_utils.read_text_from_path
```
````

````{py:function} save_text_to_path(text: str, path: elasticai.creator.file_generation.resource_utils.PathType, encoding: str = 'utf-8') -> None
:canonical: elasticai.creator.file_generation.resource_utils.save_text_to_path

```{autodoc2-docstring} elasticai.creator.file_generation.resource_utils.save_text_to_path
```
````

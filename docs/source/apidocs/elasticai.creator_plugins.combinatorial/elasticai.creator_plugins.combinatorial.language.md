# {py:mod}`elasticai.creator_plugins.combinatorial.language`

```{py:module} elasticai.creator_plugins.combinatorial.language
```

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.language
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Port <elasticai.creator_plugins.combinatorial.language.Port>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.language.Port
    :summary:
    ```
* - {py:obj}`VHDLEntity <elasticai.creator_plugins.combinatorial.language.VHDLEntity>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.language.VHDLEntity
    :summary:
    ```
````

### API

`````{py:class} Port
:canonical: elasticai.creator_plugins.combinatorial.language.Port

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.language.Port
```

````{py:attribute} inputs
:canonical: elasticai.creator_plugins.combinatorial.language.Port.inputs
:type: dict[str, str]
:value: >
   None

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.language.Port.inputs
```

````

````{py:attribute} outputs
:canonical: elasticai.creator_plugins.combinatorial.language.Port.outputs
:type: dict[str, str]
:value: >
   None

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.language.Port.outputs
```

````

````{py:method} signals() -> typing.Iterator[str]
:canonical: elasticai.creator_plugins.combinatorial.language.Port.signals

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.language.Port.signals
```

````

`````

`````{py:class} VHDLEntity(name: str, port: elasticai.creator_plugins.combinatorial.language.Port, generics: dict[str, str])
:canonical: elasticai.creator_plugins.combinatorial.language.VHDLEntity

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.language.VHDLEntity
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.language.VHDLEntity.__init__
```

````{py:method} _generate_generic()
:canonical: elasticai.creator_plugins.combinatorial.language.VHDLEntity._generate_generic

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.language.VHDLEntity._generate_generic
```

````

````{py:method} _generate_port()
:canonical: elasticai.creator_plugins.combinatorial.language.VHDLEntity._generate_port

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.language.VHDLEntity._generate_port
```

````

````{py:method} _generate_library_clause()
:canonical: elasticai.creator_plugins.combinatorial.language.VHDLEntity._generate_library_clause

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.language.VHDLEntity._generate_library_clause
```

````

````{py:method} generate_entity() -> typing.Iterator[str]
:canonical: elasticai.creator_plugins.combinatorial.language.VHDLEntity.generate_entity

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.language.VHDLEntity.generate_entity
```

````

`````

# {py:mod}`elasticai.creator.template`

```{py:module} elasticai.creator.template
```

```{autodoc2-docstring} elasticai.creator.template
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Template <elasticai.creator.template.Template>`
  -
* - {py:obj}`TemplateParameterType <elasticai.creator.template.TemplateParameterType>`
  -
* - {py:obj}`AnalysingTemplateParameterType <elasticai.creator.template.AnalysingTemplateParameterType>`
  -
* - {py:obj}`TemplateBuilder <elasticai.creator.template.TemplateBuilder>`
  - ```{autodoc2-docstring} elasticai.creator.template.TemplateBuilder
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TemplateParameter <elasticai.creator.template.TemplateParameter>`
  - ```{autodoc2-docstring} elasticai.creator.template.TemplateParameter
    :summary:
    ```
````

### API

````{py:data} TemplateParameter
:canonical: elasticai.creator.template.TemplateParameter
:type: typing.TypeAlias
:value: >
   None

```{autodoc2-docstring} elasticai.creator.template.TemplateParameter
```

````

`````{py:class} Template
:canonical: elasticai.creator.template.Template

Bases: {py:obj}`typing.Protocol`

````{py:method} render(mapping: dict[str, elasticai.creator.template.TemplateParameter]) -> str
:canonical: elasticai.creator.template.Template.render

```{autodoc2-docstring} elasticai.creator.template.Template.render
```

````

`````

`````{py:class} TemplateParameterType
:canonical: elasticai.creator.template.TemplateParameterType

Bases: {py:obj}`typing.Protocol`

````{py:attribute} regex
:canonical: elasticai.creator.template.TemplateParameterType.regex
:type: str
:value: >
   None

```{autodoc2-docstring} elasticai.creator.template.TemplateParameterType.regex
```

````

````{py:method} replace(m: re.Match[typing.AnyStr]) -> str
:canonical: elasticai.creator.template.TemplateParameterType.replace

```{autodoc2-docstring} elasticai.creator.template.TemplateParameterType.replace
```

````

`````

`````{py:class} AnalysingTemplateParameterType
:canonical: elasticai.creator.template.AnalysingTemplateParameterType

Bases: {py:obj}`elasticai.creator.template.TemplateParameterType`, {py:obj}`typing.Protocol`

````{py:attribute} analyse_regex
:canonical: elasticai.creator.template.AnalysingTemplateParameterType.analyse_regex
:type: str
:value: >
   None

```{autodoc2-docstring} elasticai.creator.template.AnalysingTemplateParameterType.analyse_regex
```

````

````{py:method} analyse(m: re.Match) -> None
:canonical: elasticai.creator.template.AnalysingTemplateParameterType.analyse

```{autodoc2-docstring} elasticai.creator.template.AnalysingTemplateParameterType.analyse
```

````

`````

`````{py:class} TemplateBuilder()
:canonical: elasticai.creator.template.TemplateBuilder

```{autodoc2-docstring} elasticai.creator.template.TemplateBuilder
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.template.TemplateBuilder.__init__
```

````{py:method} set_prototype(prototype: str | tuple[str, ...] | list[str]) -> elasticai.creator.template.TemplateBuilder
:canonical: elasticai.creator.template.TemplateBuilder.set_prototype

```{autodoc2-docstring} elasticai.creator.template.TemplateBuilder.set_prototype
```

````

````{py:method} add_parameter(name: str, _type: elasticai.creator.template.TemplateParameterType) -> elasticai.creator.template.TemplateBuilder
:canonical: elasticai.creator.template.TemplateBuilder.add_parameter

```{autodoc2-docstring} elasticai.creator.template.TemplateBuilder.add_parameter
```

````

````{py:method} build() -> elasticai.creator.template.Template
:canonical: elasticai.creator.template.TemplateBuilder.build

```{autodoc2-docstring} elasticai.creator.template.TemplateBuilder.build
```

````

`````

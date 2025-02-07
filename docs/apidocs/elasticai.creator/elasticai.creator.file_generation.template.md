# {py:mod}`elasticai.creator.file_generation.template`

```{py:module} elasticai.creator.file_generation.template
```

```{autodoc2-docstring} elasticai.creator.file_generation.template
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Template <elasticai.creator.file_generation.template.Template>`
  -
* - {py:obj}`InProjectTemplate <elasticai.creator.file_generation.template.InProjectTemplate>`
  -
* - {py:obj}`TemplateExpander <elasticai.creator.file_generation.template.TemplateExpander>`
  - ```{autodoc2-docstring} elasticai.creator.file_generation.template.TemplateExpander
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`module_to_package <elasticai.creator.file_generation.template.module_to_package>`
  - ```{autodoc2-docstring} elasticai.creator.file_generation.template.module_to_package
    :summary:
    ```
````

### API

````{py:function} module_to_package(module: str) -> str
:canonical: elasticai.creator.file_generation.template.module_to_package

```{autodoc2-docstring} elasticai.creator.file_generation.template.module_to_package
```
````

`````{py:class} Template
:canonical: elasticai.creator.file_generation.template.Template

Bases: {py:obj}`typing.Protocol`

````{py:attribute} parameters
:canonical: elasticai.creator.file_generation.template.Template.parameters
:type: dict[str, str | list[str]]
:value: >
   None

```{autodoc2-docstring} elasticai.creator.file_generation.template.Template.parameters
```

````

````{py:attribute} content
:canonical: elasticai.creator.file_generation.template.Template.content
:type: list[str]
:value: >
   None

```{autodoc2-docstring} elasticai.creator.file_generation.template.Template.content
```

````

`````

`````{py:class} InProjectTemplate
:canonical: elasticai.creator.file_generation.template.InProjectTemplate

Bases: {py:obj}`elasticai.creator.file_generation.template.Template`

````{py:attribute} package
:canonical: elasticai.creator.file_generation.template.InProjectTemplate.package
:type: str
:value: >
   None

```{autodoc2-docstring} elasticai.creator.file_generation.template.InProjectTemplate.package
```

````

````{py:attribute} file_name
:canonical: elasticai.creator.file_generation.template.InProjectTemplate.file_name
:type: str
:value: >
   None

```{autodoc2-docstring} elasticai.creator.file_generation.template.InProjectTemplate.file_name
```

````

````{py:attribute} parameters
:canonical: elasticai.creator.file_generation.template.InProjectTemplate.parameters
:type: dict[str, str | list[str]]
:value: >
   None

```{autodoc2-docstring} elasticai.creator.file_generation.template.InProjectTemplate.parameters
```

````

````{py:method} __post_init__() -> None
:canonical: elasticai.creator.file_generation.template.InProjectTemplate.__post_init__

```{autodoc2-docstring} elasticai.creator.file_generation.template.InProjectTemplate.__post_init__
```

````

`````

`````{py:class} TemplateExpander(template: elasticai.creator.file_generation.template.Template)
:canonical: elasticai.creator.file_generation.template.TemplateExpander

```{autodoc2-docstring} elasticai.creator.file_generation.template.TemplateExpander
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.file_generation.template.TemplateExpander.__init__
```

````{py:method} lines() -> list[str]
:canonical: elasticai.creator.file_generation.template.TemplateExpander.lines

```{autodoc2-docstring} elasticai.creator.file_generation.template.TemplateExpander.lines
```

````

````{py:method} unfilled_variables() -> set[str]
:canonical: elasticai.creator.file_generation.template.TemplateExpander.unfilled_variables

```{autodoc2-docstring} elasticai.creator.file_generation.template.TemplateExpander.unfilled_variables
```

````

`````

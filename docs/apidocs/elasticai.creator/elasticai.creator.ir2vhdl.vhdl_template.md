# {py:mod}`elasticai.creator.ir2vhdl.vhdl_template`

```{py:module} elasticai.creator.ir2vhdl.vhdl_template
```

```{autodoc2-docstring} elasticai.creator.ir2vhdl.vhdl_template
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EntityTemplateParameter <elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateParameter>`
  - ```{autodoc2-docstring} elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateParameter
    :summary:
    ```
* - {py:obj}`ValueTemplateParameter <elasticai.creator.ir2vhdl.vhdl_template.ValueTemplateParameter>`
  - ```{autodoc2-docstring} elasticai.creator.ir2vhdl.vhdl_template.ValueTemplateParameter
    :summary:
    ```
* - {py:obj}`EntityTemplateDirector <elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateDirector>`
  - ```{autodoc2-docstring} elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateDirector
    :summary:
    ```
````

### API

`````{py:class} EntityTemplateParameter()
:canonical: elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateParameter

Bases: {py:obj}`elasticai.creator.template.AnalysingTemplateParameterType`

```{autodoc2-docstring} elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateParameter
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateParameter.__init__
```

````{py:method} analyse(m: re.Match) -> None
:canonical: elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateParameter.analyse

```{autodoc2-docstring} elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateParameter.analyse
```

````

````{py:method} replace(m: re.Match)
:canonical: elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateParameter.replace

```{autodoc2-docstring} elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateParameter.replace
```

````

`````

`````{py:class} ValueTemplateParameter()
:canonical: elasticai.creator.ir2vhdl.vhdl_template.ValueTemplateParameter

Bases: {py:obj}`elasticai.creator.template.TemplateParameterType`

```{autodoc2-docstring} elasticai.creator.ir2vhdl.vhdl_template.ValueTemplateParameter
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.ir2vhdl.vhdl_template.ValueTemplateParameter.__init__
```

````{py:method} replace(m: re.Match) -> str
:canonical: elasticai.creator.ir2vhdl.vhdl_template.ValueTemplateParameter.replace

```{autodoc2-docstring} elasticai.creator.ir2vhdl.vhdl_template.ValueTemplateParameter.replace
```

````

`````

`````{py:class} EntityTemplateDirector()
:canonical: elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateDirector

```{autodoc2-docstring} elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateDirector
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateDirector.__init__
```

````{py:method} set_prototype(prototype: str) -> elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateDirector
:canonical: elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateDirector.set_prototype

```{autodoc2-docstring} elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateDirector.set_prototype
```

````

````{py:method} add_generic(name: str) -> elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateDirector
:canonical: elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateDirector.add_generic

```{autodoc2-docstring} elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateDirector.add_generic
```

````

````{py:method} build() -> elasticai.creator.template.Template
:canonical: elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateDirector.build

```{autodoc2-docstring} elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateDirector.build
```

````

`````

# {py:mod}`elasticai.creator.vhdl.design.design`

```{py:module} elasticai.creator.vhdl.design.design
```

```{autodoc2-docstring} elasticai.creator.vhdl.design.design
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Design <elasticai.creator.vhdl.design.design.Design>`
  -
````

### API

`````{py:class} Design(name: str)
:canonical: elasticai.creator.vhdl.design.design.Design

Bases: {py:obj}`elasticai.creator.file_generation.savable.Savable`, {py:obj}`abc.ABC`

````{py:property} name
:canonical: elasticai.creator.vhdl.design.design.Design.name
:type: str

```{autodoc2-docstring} elasticai.creator.vhdl.design.design.Design.name
```

````

````{py:property} port
:canonical: elasticai.creator.vhdl.design.design.Design.port
:abstractmethod:
:type: elasticai.creator.vhdl.design.ports.Port

```{autodoc2-docstring} elasticai.creator.vhdl.design.design.Design.port
```

````

````{py:method} save_to(destination: elasticai.creator.file_generation.savable.Path) -> None
:canonical: elasticai.creator.vhdl.design.design.Design.save_to
:abstractmethod:

```{autodoc2-docstring} elasticai.creator.vhdl.design.design.Design.save_to
```

````

`````

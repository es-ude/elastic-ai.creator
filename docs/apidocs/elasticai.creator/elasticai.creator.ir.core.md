# {py:mod}`elasticai.creator.ir.core`

```{py:module} elasticai.creator.ir.core
```

```{autodoc2-docstring} elasticai.creator.ir.core
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Node <elasticai.creator.ir.core.Node>`
  - ```{autodoc2-docstring} elasticai.creator.ir.core.Node
    :summary:
    ```
* - {py:obj}`Edge <elasticai.creator.ir.core.Edge>`
  - ```{autodoc2-docstring} elasticai.creator.ir.core.Edge
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`node <elasticai.creator.ir.core.node>`
  - ```{autodoc2-docstring} elasticai.creator.ir.core.node
    :summary:
    ```
* - {py:obj}`edge <elasticai.creator.ir.core.edge>`
  - ```{autodoc2-docstring} elasticai.creator.ir.core.edge
    :summary:
    ```
````

### API

`````{py:class} Node(data: dict[str, elasticai.creator.ir.attribute.Attribute])
:canonical: elasticai.creator.ir.core.Node

Bases: {py:obj}`elasticai.creator.ir.ir_data.IrData`

```{autodoc2-docstring} elasticai.creator.ir.core.Node
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.ir.core.Node.__init__
```

````{py:attribute} name
:canonical: elasticai.creator.ir.core.Node.name
:type: elasticai.creator.ir.required_field.ReadOnlyField[str, str]
:value: >
   '_read_only_str(...)'

```{autodoc2-docstring} elasticai.creator.ir.core.Node.name
```

````

````{py:attribute} type
:canonical: elasticai.creator.ir.core.Node.type
:type: elasticai.creator.ir.required_field.ReadOnlyField[str, str]
:value: >
   '_read_only_str(...)'

```{autodoc2-docstring} elasticai.creator.ir.core.Node.type
```

````

`````

`````{py:class} Edge(data: dict[str, elasticai.creator.ir.attribute.Attribute])
:canonical: elasticai.creator.ir.core.Edge

Bases: {py:obj}`elasticai.creator.ir.ir_data.IrData`

```{autodoc2-docstring} elasticai.creator.ir.core.Edge
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.ir.core.Edge.__init__
```

````{py:attribute} src
:canonical: elasticai.creator.ir.core.Edge.src
:type: elasticai.creator.ir.required_field.ReadOnlyField[str, str]
:value: >
   '_read_only_str(...)'

```{autodoc2-docstring} elasticai.creator.ir.core.Edge.src
```

````

````{py:attribute} sink
:canonical: elasticai.creator.ir.core.Edge.sink
:type: elasticai.creator.ir.required_field.ReadOnlyField[str, str]
:value: >
   '_read_only_str(...)'

```{autodoc2-docstring} elasticai.creator.ir.core.Edge.sink
```

````

`````

````{py:function} node(name: str, type: str, attributes: dict[str, elasticai.creator.ir.attribute.Attribute] | None = None) -> elasticai.creator.ir.core.Node
:canonical: elasticai.creator.ir.core.node

```{autodoc2-docstring} elasticai.creator.ir.core.node
```
````

````{py:function} edge(src: str, sink: str, attributes: dict[str, elasticai.creator.ir.attribute.Attribute] | None = None) -> elasticai.creator.ir.core.Edge
:canonical: elasticai.creator.ir.core.edge

```{autodoc2-docstring} elasticai.creator.ir.core.edge
```
````

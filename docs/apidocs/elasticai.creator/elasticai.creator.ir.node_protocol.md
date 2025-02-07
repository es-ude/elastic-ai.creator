# {py:mod}`elasticai.creator.ir.node_protocol`

```{py:module} elasticai.creator.ir.node_protocol
```

```{autodoc2-docstring} elasticai.creator.ir.node_protocol
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Node <elasticai.creator.ir.node_protocol.Node>`
  -
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NodeT <elasticai.creator.ir.node_protocol.NodeT>`
  - ```{autodoc2-docstring} elasticai.creator.ir.node_protocol.NodeT
    :summary:
    ```
````

### API

`````{py:class} Node
:canonical: elasticai.creator.ir.node_protocol.Node

Bases: {py:obj}`typing.Protocol`

````{py:attribute} name
:canonical: elasticai.creator.ir.node_protocol.Node.name
:type: str
:value: >
   None

```{autodoc2-docstring} elasticai.creator.ir.node_protocol.Node.name
```

````

````{py:attribute} type
:canonical: elasticai.creator.ir.node_protocol.Node.type
:type: str
:value: >
   None

```{autodoc2-docstring} elasticai.creator.ir.node_protocol.Node.type
```

````

````{py:attribute} data
:canonical: elasticai.creator.ir.node_protocol.Node.data
:type: dict[str, elasticai.creator.ir.attribute.Attribute]
:value: >
   None

```{autodoc2-docstring} elasticai.creator.ir.node_protocol.Node.data
```

````

````{py:attribute} attributes
:canonical: elasticai.creator.ir.node_protocol.Node.attributes
:type: dict[str, elasticai.creator.ir.attribute.Attribute]
:value: >
   None

```{autodoc2-docstring} elasticai.creator.ir.node_protocol.Node.attributes
```

````

````{py:method} new(*args, **kwargs) -> typing_extensions.Self
:canonical: elasticai.creator.ir.node_protocol.Node.new
:classmethod:

```{autodoc2-docstring} elasticai.creator.ir.node_protocol.Node.new
```

````

````{py:method} from_dict(d: dict[str, elasticai.creator.ir.attribute.Attribute]) -> typing_extensions.Self
:canonical: elasticai.creator.ir.node_protocol.Node.from_dict
:classmethod:

```{autodoc2-docstring} elasticai.creator.ir.node_protocol.Node.from_dict
```

````

````{py:method} as_dict() -> dict[str, elasticai.creator.ir.attribute.Attribute]
:canonical: elasticai.creator.ir.node_protocol.Node.as_dict

```{autodoc2-docstring} elasticai.creator.ir.node_protocol.Node.as_dict
```

````

`````

````{py:data} NodeT
:canonical: elasticai.creator.ir.node_protocol.NodeT
:value: >
   'TypeVar(...)'

```{autodoc2-docstring} elasticai.creator.ir.node_protocol.NodeT
```

````

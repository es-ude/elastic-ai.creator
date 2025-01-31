# {py:mod}`elasticai.creator.torch2ir`

```{py:module} elasticai.creator.torch2ir
```

```{autodoc2-docstring} elasticai.creator.torch2ir
:allowtitles:
```

## Package Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Torch2Ir <elasticai.creator.torch2ir.torch2ir.Torch2Ir>`
  - ```{autodoc2-docstring} elasticai.creator.torch2ir.torch2ir.Torch2Ir
    :summary:
    ```
* - {py:obj}`Implementation <elasticai.creator.torch2ir.core.Implementation>`
  -
* - {py:obj}`Node <elasticai.creator.torch2ir.core.Node>`
  -
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`new_node <elasticai.creator.torch2ir.core.new_node>`
  - ```{autodoc2-docstring} elasticai.creator.torch2ir.core.new_node
    :summary:
    ```
* - {py:obj}`input_node <elasticai.creator.torch2ir.core.input_node>`
  - ```{autodoc2-docstring} elasticai.creator.torch2ir.core.input_node
    :summary:
    ```
* - {py:obj}`output_node <elasticai.creator.torch2ir.core.output_node>`
  - ```{autodoc2-docstring} elasticai.creator.torch2ir.core.output_node
    :summary:
    ```
````

### API

`````{py:class} Torch2Ir(tracer: torch.fx.Tracer = _DefaultTracer())
:canonical: elasticai.creator.torch2ir.torch2ir.Torch2Ir

```{autodoc2-docstring} elasticai.creator.torch2ir.torch2ir.Torch2Ir
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.torch2ir.torch2ir.Torch2Ir.__init__
```

````{py:method} register_module_handler(module_type: str, handler: collections.abc.Callable[[torch.nn.Module], dict]) -> None
:canonical: elasticai.creator.torch2ir.torch2ir.Torch2Ir.register_module_handler

```{autodoc2-docstring} elasticai.creator.torch2ir.torch2ir.Torch2Ir.register_module_handler
```

````

````{py:method} register_module_handlers(handlers: collections.abc.Iterable[collections.abc.Callable[[torch.nn.Module], dict]]) -> elasticai.creator.torch2ir.torch2ir.Torch2Ir
:canonical: elasticai.creator.torch2ir.torch2ir.Torch2Ir.register_module_handlers

```{autodoc2-docstring} elasticai.creator.torch2ir.torch2ir.Torch2Ir.register_module_handlers
```

````

````{py:method} convert(model: torch.nn.Module) -> dict[str, elasticai.creator.torch2ir.core.Implementation]
:canonical: elasticai.creator.torch2ir.torch2ir.Torch2Ir.convert

```{autodoc2-docstring} elasticai.creator.torch2ir.torch2ir.Torch2Ir.convert
```

````

````{py:method} get_default_converter() -> elasticai.creator.torch2ir.torch2ir.Torch2Ir
:canonical: elasticai.creator.torch2ir.torch2ir.Torch2Ir.get_default_converter
:classmethod:

```{autodoc2-docstring} elasticai.creator.torch2ir.torch2ir.Torch2Ir.get_default_converter
```

````

`````

`````{py:class} Implementation(*, node_fn: collections.abc.Callable[[dict], elasticai.creator.ir.graph.N] = Node, edge_fn: collections.abc.Callable[[dict], elasticai.creator.ir.graph.E] = Edge, nodes: collections.abc.Iterable[elasticai.creator.ir.graph.N] = tuple(), edges: collections.abc.Iterable[elasticai.creator.ir.graph.E] = tuple(), data=None)
:canonical: elasticai.creator.torch2ir.core.Implementation

Bases: {py:obj}`elasticai.creator.ir.Graph`\[{py:obj}`elasticai.creator.torch2ir.core.Node`\, {py:obj}`elasticai.creator.ir.Edge`\]

````{py:attribute} name
:canonical: elasticai.creator.torch2ir.core.Implementation.name
:type: str
:value: >
   None

```{autodoc2-docstring} elasticai.creator.torch2ir.core.Implementation.name
```

````

````{py:attribute} type
:canonical: elasticai.creator.torch2ir.core.Implementation.type
:type: str
:value: >
   None

```{autodoc2-docstring} elasticai.creator.torch2ir.core.Implementation.type
```

````

`````

`````{py:class} Node(data: dict[str, elasticai.creator.ir.attribute.Attribute])
:canonical: elasticai.creator.torch2ir.core.Node

Bases: {py:obj}`elasticai.creator.ir.Node`

````{py:attribute} implementation
:canonical: elasticai.creator.torch2ir.core.Node.implementation
:type: str
:value: >
   None

```{autodoc2-docstring} elasticai.creator.torch2ir.core.Node.implementation
```

````

`````

````{py:function} new_node(name: str, type: str, implementation: str, attributes: dict[str, typing.Any] | None = None) -> elasticai.creator.torch2ir.core.Node
:canonical: elasticai.creator.torch2ir.core.new_node

```{autodoc2-docstring} elasticai.creator.torch2ir.core.new_node
```
````

````{py:function} input_node(attributes: dict[str, typing.Any] | None = None) -> elasticai.creator.torch2ir.core.Node
:canonical: elasticai.creator.torch2ir.core.input_node

```{autodoc2-docstring} elasticai.creator.torch2ir.core.input_node
```
````

````{py:function} output_node(attributes: dict[str, typing.Any] | None = None) -> elasticai.creator.torch2ir.core.Node
:canonical: elasticai.creator.torch2ir.core.output_node

```{autodoc2-docstring} elasticai.creator.torch2ir.core.output_node
```
````

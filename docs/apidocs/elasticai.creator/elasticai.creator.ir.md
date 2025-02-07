# {py:mod}`elasticai.creator.ir`

```{py:module} elasticai.creator.ir
```

```{autodoc2-docstring} elasticai.creator.ir
:allowtitles:
```

## Package Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RequiredField <elasticai.creator.ir.required_field.RequiredField>`
  - ```{autodoc2-docstring} elasticai.creator.ir.required_field.RequiredField
    :summary:
    ```
* - {py:obj}`SimpleRequiredField <elasticai.creator.ir.required_field.SimpleRequiredField>`
  -
* - {py:obj}`IrData <elasticai.creator.ir.ir_data.IrData>`
  - ```{autodoc2-docstring} elasticai.creator.ir.ir_data.IrData
    :summary:
    ```
* - {py:obj}`IrDataMeta <elasticai.creator.ir.ir_data_meta.IrDataMeta>`
  - ```{autodoc2-docstring} elasticai.creator.ir.ir_data_meta.IrDataMeta
    :summary:
    ```
* - {py:obj}`Edge <elasticai.creator.ir.core.Edge>`
  - ```{autodoc2-docstring} elasticai.creator.ir.core.Edge
    :summary:
    ```
* - {py:obj}`Node <elasticai.creator.ir.core.Node>`
  - ```{autodoc2-docstring} elasticai.creator.ir.core.Node
    :summary:
    ```
* - {py:obj}`LoweringPass <elasticai.creator.ir.lowering.LoweringPass>`
  -
* - {py:obj}`Lowerable <elasticai.creator.ir.lowering.Lowerable>`
  -
* - {py:obj}`Graph <elasticai.creator.ir.graph.Graph>`
  -
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`edge <elasticai.creator.ir.core.edge>`
  - ```{autodoc2-docstring} elasticai.creator.ir.core.edge
    :summary:
    ```
* - {py:obj}`node <elasticai.creator.ir.core.node>`
  - ```{autodoc2-docstring} elasticai.creator.ir.core.node
    :summary:
    ```
````

### API

`````{py:class} RequiredField(set_convert: collections.abc.Callable[[elasticai.creator.ir.required_field.VisibleT], elasticai.creator.ir.required_field.StoredT], get_convert: collections.abc.Callable[[elasticai.creator.ir.required_field.StoredT], elasticai.creator.ir.required_field.VisibleT])
:canonical: elasticai.creator.ir.required_field.RequiredField

Bases: {py:obj}`typing.Generic`\[{py:obj}`elasticai.creator.ir.required_field.StoredT`\, {py:obj}`elasticai.creator.ir.required_field.VisibleT`\]

```{autodoc2-docstring} elasticai.creator.ir.required_field.RequiredField
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.ir.required_field.RequiredField.__init__
```

````{py:attribute} __slots__
:canonical: elasticai.creator.ir.required_field.RequiredField.__slots__
:value: >
   ('set_convert', 'get_convert', 'name')

```{autodoc2-docstring} elasticai.creator.ir.required_field.RequiredField.__slots__
```

````

````{py:method} __set_name__(owner: type[elasticai.creator.ir._has_data.HasData], name: str) -> None
:canonical: elasticai.creator.ir.required_field.RequiredField.__set_name__

```{autodoc2-docstring} elasticai.creator.ir.required_field.RequiredField.__set_name__
```

````

````{py:method} __get__(instance: elasticai.creator.ir._has_data.HasData, owner=None) -> elasticai.creator.ir.required_field.VisibleT
:canonical: elasticai.creator.ir.required_field.RequiredField.__get__

```{autodoc2-docstring} elasticai.creator.ir.required_field.RequiredField.__get__
```

````

````{py:method} __set__(instance: elasticai.creator.ir._has_data.HasData, value: elasticai.creator.ir.required_field.VisibleT) -> None
:canonical: elasticai.creator.ir.required_field.RequiredField.__set__

```{autodoc2-docstring} elasticai.creator.ir.required_field.RequiredField.__set__
```

````

`````

`````{py:class} SimpleRequiredField()
:canonical: elasticai.creator.ir.required_field.SimpleRequiredField

Bases: {py:obj}`elasticai.creator.ir.required_field.RequiredField`\[{py:obj}`elasticai.creator.ir.required_field.StoredT`\, {py:obj}`elasticai.creator.ir.required_field.StoredT`\]

````{py:attribute} slots
:canonical: elasticai.creator.ir.required_field.SimpleRequiredField.slots
:value: >
   ('get_convert', 'set_convert', 'name')

```{autodoc2-docstring} elasticai.creator.ir.required_field.SimpleRequiredField.slots
```

````

`````

`````{py:class} IrData(data: dict[str, elasticai.creator.ir.attribute.Attribute])
:canonical: elasticai.creator.ir.ir_data.IrData

```{autodoc2-docstring} elasticai.creator.ir.ir_data.IrData
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.ir.ir_data.IrData.__init__
```

````{py:attribute} attributes
:canonical: elasticai.creator.ir.ir_data.IrData.attributes
:type: elasticai.creator.ir.attributes_descriptor.AttributesDescriptor
:value: >
   'AttributesDescriptor(...)'

```{autodoc2-docstring} elasticai.creator.ir.ir_data.IrData.attributes
```

````

````{py:method} get_missing_required_fields() -> dict[str, type]
:canonical: elasticai.creator.ir.ir_data.IrData.get_missing_required_fields

```{autodoc2-docstring} elasticai.creator.ir.ir_data.IrData.get_missing_required_fields
```

````

````{py:method} __repr__() -> str
:canonical: elasticai.creator.ir.ir_data.IrData.__repr__

````

````{py:method} __eq__(o: object) -> bool
:canonical: elasticai.creator.ir.ir_data.IrData.__eq__

````

`````

`````{py:class} IrDataMeta
:canonical: elasticai.creator.ir.ir_data_meta.IrDataMeta

Bases: {py:obj}`type`

```{autodoc2-docstring} elasticai.creator.ir.ir_data_meta.IrDataMeta
```

````{py:property} required_fields
:canonical: elasticai.creator.ir.ir_data_meta.IrDataMeta.required_fields
:type: types.MappingProxyType[str, type]

```{autodoc2-docstring} elasticai.creator.ir.ir_data_meta.IrDataMeta.required_fields
```

````

````{py:method} __prepare__(name, bases, **kwds)
:canonical: elasticai.creator.ir.ir_data_meta.IrDataMeta.__prepare__
:classmethod:

```{autodoc2-docstring} elasticai.creator.ir.ir_data_meta.IrDataMeta.__prepare__
```

````

````{py:method} __new__(name: str, bases: tuple[type, ...], namespace: dict[str, typing.Any], **kwds) -> elasticai.creator.ir.ir_data_meta.IrDataMeta
:canonical: elasticai.creator.ir.ir_data_meta.IrDataMeta.__new__

```{autodoc2-docstring} elasticai.creator.ir.ir_data_meta.IrDataMeta.__new__
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

````{py:function} edge(src: str, sink: str, attributes: dict[str, elasticai.creator.ir.attribute.Attribute] | None = None) -> elasticai.creator.ir.core.Edge
:canonical: elasticai.creator.ir.core.edge

```{autodoc2-docstring} elasticai.creator.ir.core.edge
```
````

````{py:function} node(name: str, type: str, attributes: dict[str, elasticai.creator.ir.attribute.Attribute] | None = None) -> elasticai.creator.ir.core.Node
:canonical: elasticai.creator.ir.core.node

```{autodoc2-docstring} elasticai.creator.ir.core.node
```
````

`````{py:class} LoweringPass()
:canonical: elasticai.creator.ir.lowering.LoweringPass

Bases: {py:obj}`typing.Generic`\[{py:obj}`elasticai.creator.ir.lowering.Tin`\, {py:obj}`elasticai.creator.ir.lowering.Tout`\]

````{py:attribute} register
:canonical: elasticai.creator.ir.lowering.LoweringPass.register
:type: elasticai.creator.function_utils.RegisterDescriptor[elasticai.creator.ir.lowering.Tin, elasticai.creator.ir.lowering.Tout]
:value: >
   'RegisterDescriptor(...)'

```{autodoc2-docstring} elasticai.creator.ir.lowering.LoweringPass.register
```

````

````{py:attribute} register_iterable
:canonical: elasticai.creator.ir.lowering.LoweringPass.register_iterable
:type: elasticai.creator.function_utils.RegisterDescriptor[elasticai.creator.ir.lowering.Tin, collections.abc.Iterable[elasticai.creator.ir.lowering.Tout]]
:value: >
   'RegisterDescriptor(...)'

```{autodoc2-docstring} elasticai.creator.ir.lowering.LoweringPass.register_iterable
```

````

````{py:method} __call__(args: collections.abc.Iterable[elasticai.creator.ir.lowering.Tin]) -> collections.abc.Iterator[elasticai.creator.ir.lowering.Tout]
:canonical: elasticai.creator.ir.lowering.LoweringPass.__call__

```{autodoc2-docstring} elasticai.creator.ir.lowering.LoweringPass.__call__
```

````

`````

`````{py:class} Lowerable
:canonical: elasticai.creator.ir.lowering.Lowerable

Bases: {py:obj}`typing.Protocol`

````{py:property} type
:canonical: elasticai.creator.ir.lowering.Lowerable.type
:abstractmethod:
:type: str

```{autodoc2-docstring} elasticai.creator.ir.lowering.Lowerable.type
```

````

`````

`````{py:class} Graph(nodes: collections.abc.Iterable[elasticai.creator.ir.graph.N] = tuple(), edges: collections.abc.Iterable[elasticai.creator.ir.graph.E] = tuple())
:canonical: elasticai.creator.ir.graph.Graph

Bases: {py:obj}`typing.Generic`\[{py:obj}`elasticai.creator.ir.graph.N`\, {py:obj}`elasticai.creator.ir.graph.E`\]

````{py:method} add_node(n: elasticai.creator.ir.graph.N) -> None
:canonical: elasticai.creator.ir.graph.Graph.add_node

```{autodoc2-docstring} elasticai.creator.ir.graph.Graph.add_node
```

````

````{py:method} add_nodes(ns: collections.abc.Iterable[elasticai.creator.ir.graph.N]) -> None
:canonical: elasticai.creator.ir.graph.Graph.add_nodes

```{autodoc2-docstring} elasticai.creator.ir.graph.Graph.add_nodes
```

````

````{py:method} add_edges(es: collections.abc.Iterable[elasticai.creator.ir.graph.E]) -> None
:canonical: elasticai.creator.ir.graph.Graph.add_edges

```{autodoc2-docstring} elasticai.creator.ir.graph.Graph.add_edges
```

````

````{py:method} add_edge(e: elasticai.creator.ir.graph.E) -> None
:canonical: elasticai.creator.ir.graph.Graph.add_edge

```{autodoc2-docstring} elasticai.creator.ir.graph.Graph.add_edge
```

````

````{py:method} successors(node: str | elasticai.creator.ir.graph.N) -> collections.abc.Mapping[str, elasticai.creator.ir.graph.N]
:canonical: elasticai.creator.ir.graph.Graph.successors

```{autodoc2-docstring} elasticai.creator.ir.graph.Graph.successors
```

````

````{py:method} predecessors(node: str | elasticai.creator.ir.graph.N) -> collections.abc.Mapping[str, elasticai.creator.ir.graph.N]
:canonical: elasticai.creator.ir.graph.Graph.predecessors

```{autodoc2-docstring} elasticai.creator.ir.graph.Graph.predecessors
```

````

````{py:property} nodes
:canonical: elasticai.creator.ir.graph.Graph.nodes
:type: collections.abc.Mapping[str, elasticai.creator.ir.graph.N]

```{autodoc2-docstring} elasticai.creator.ir.graph.Graph.nodes
```

````

````{py:property} edges
:canonical: elasticai.creator.ir.graph.Graph.edges
:type: collections.abc.Mapping[tuple[str, str], elasticai.creator.ir.graph.E]

```{autodoc2-docstring} elasticai.creator.ir.graph.Graph.edges
```

````

````{py:method} iter_bfs_down_from(node: str) -> collections.abc.Iterator[elasticai.creator.ir.graph.N]
:canonical: elasticai.creator.ir.graph.Graph.iter_bfs_down_from

```{autodoc2-docstring} elasticai.creator.ir.graph.Graph.iter_bfs_down_from
```

````

````{py:method} iter_bfs_up_from(node: str) -> collections.abc.Iterator[elasticai.creator.ir.graph.N]
:canonical: elasticai.creator.ir.graph.Graph.iter_bfs_up_from

```{autodoc2-docstring} elasticai.creator.ir.graph.Graph.iter_bfs_up_from
```

````

````{py:method} iter_dfs_preorder_down_from(node: str) -> collections.abc.Iterator[elasticai.creator.ir.graph.N]
:canonical: elasticai.creator.ir.graph.Graph.iter_dfs_preorder_down_from

```{autodoc2-docstring} elasticai.creator.ir.graph.Graph.iter_dfs_preorder_down_from
```

````

`````

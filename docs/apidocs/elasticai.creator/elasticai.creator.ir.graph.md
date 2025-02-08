# {py:mod}`elasticai.creator.ir.graph`

```{py:module} elasticai.creator.ir.graph
```

```{autodoc2-docstring} elasticai.creator.ir.graph
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Graph <elasticai.creator.ir.graph.Graph>`
  -
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`N <elasticai.creator.ir.graph.N>`
  - ```{autodoc2-docstring} elasticai.creator.ir.graph.N
    :summary:
    ```
* - {py:obj}`E <elasticai.creator.ir.graph.E>`
  - ```{autodoc2-docstring} elasticai.creator.ir.graph.E
    :summary:
    ```
````

### API

````{py:data} N
:canonical: elasticai.creator.ir.graph.N
:value: >
   'TypeVar(...)'

```{autodoc2-docstring} elasticai.creator.ir.graph.N
```

````

````{py:data} E
:canonical: elasticai.creator.ir.graph.E
:value: >
   'TypeVar(...)'

```{autodoc2-docstring} elasticai.creator.ir.graph.E
```

````

`````{py:class} Graph(*, node_fn: collections.abc.Callable[[dict], elasticai.creator.ir.graph.N] = Node, edge_fn: collections.abc.Callable[[dict], elasticai.creator.ir.graph.E] = Edge, nodes: collections.abc.Iterable[elasticai.creator.ir.graph.N] = tuple(), edges: collections.abc.Iterable[elasticai.creator.ir.graph.E] = tuple(), data=None)
:canonical: elasticai.creator.ir.graph.Graph

Bases: {py:obj}`elasticai.creator.ir.ir_data.IrData`, {py:obj}`typing.Generic`\[{py:obj}`elasticai.creator.ir.graph.N`\, {py:obj}`elasticai.creator.ir.graph.E`\]

````{py:attribute} __slots__
:canonical: elasticai.creator.ir.graph.Graph.__slots__
:value: >
   ('data', '_g', '_node_data', '_edge_data', '_node_fn', '_edge_fn')

```{autodoc2-docstring} elasticai.creator.ir.graph.Graph.__slots__
```

````

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

````{py:method} iter_bfs_down_from(node: str) -> collections.abc.Mapping[str, elasticai.creator.ir.graph.N]
:canonical: elasticai.creator.ir.graph.Graph.iter_bfs_down_from

```{autodoc2-docstring} elasticai.creator.ir.graph.Graph.iter_bfs_down_from
```

````

````{py:method} iter_bfs_up_from(node: str) -> collections.abc.Mapping[str, elasticai.creator.ir.graph.N]
:canonical: elasticai.creator.ir.graph.Graph.iter_bfs_up_from

```{autodoc2-docstring} elasticai.creator.ir.graph.Graph.iter_bfs_up_from
```

````

````{py:method} iter_dfs_preorder_down_from(node: str) -> collections.abc.Mapping[str, elasticai.creator.ir.graph.N]
:canonical: elasticai.creator.ir.graph.Graph.iter_dfs_preorder_down_from

```{autodoc2-docstring} elasticai.creator.ir.graph.Graph.iter_dfs_preorder_down_from
```

````

````{py:method} as_dict() -> dict
:canonical: elasticai.creator.ir.graph.Graph.as_dict

```{autodoc2-docstring} elasticai.creator.ir.graph.Graph.as_dict
```

````

````{py:method} from_dict(d: dict[str, typing.Any], node_fn: collections.abc.Callable[[dict], elasticai.creator.ir.graph.N] = Node, edge_fn: collections.abc.Callable[[dict], elasticai.creator.ir.graph.E] = Edge) -> Graph[N, E]
:canonical: elasticai.creator.ir.graph.Graph.from_dict
:classmethod:

```{autodoc2-docstring} elasticai.creator.ir.graph.Graph.from_dict
```

````

````{py:method} load_dict(data: dict[str, typing.Any]) -> None
:canonical: elasticai.creator.ir.graph.Graph.load_dict

```{autodoc2-docstring} elasticai.creator.ir.graph.Graph.load_dict
```

````

`````

# {py:mod}`elasticai.creator.ir.graph_delegate`

```{py:module} elasticai.creator.ir.graph_delegate
```

```{autodoc2-docstring} elasticai.creator.ir.graph_delegate
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GraphDelegate <elasticai.creator.ir.graph_delegate.GraphDelegate>`
  - ```{autodoc2-docstring} elasticai.creator.ir.graph_delegate.GraphDelegate
    :summary:
    ```
````

### API

`````{py:class} GraphDelegate()
:canonical: elasticai.creator.ir.graph_delegate.GraphDelegate

Bases: {py:obj}`typing.Generic`\[{py:obj}`elasticai.creator.ir.graph_delegate._T`\]

```{autodoc2-docstring} elasticai.creator.ir.graph_delegate.GraphDelegate
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.ir.graph_delegate.GraphDelegate.__init__
```

````{py:method} from_dict(d: dict[elasticai.creator.ir.graph_delegate._T, collections.abc.Iterable[elasticai.creator.ir.graph_delegate._T]])
:canonical: elasticai.creator.ir.graph_delegate.GraphDelegate.from_dict
:staticmethod:

```{autodoc2-docstring} elasticai.creator.ir.graph_delegate.GraphDelegate.from_dict
```

````

````{py:method} as_dict() -> dict[elasticai.creator.ir.graph_delegate._T, set[elasticai.creator.ir.graph_delegate._T]]
:canonical: elasticai.creator.ir.graph_delegate.GraphDelegate.as_dict

```{autodoc2-docstring} elasticai.creator.ir.graph_delegate.GraphDelegate.as_dict
```

````

````{py:method} add_edge(_from: elasticai.creator.ir.graph_delegate._T, _to: elasticai.creator.ir.graph_delegate._T)
:canonical: elasticai.creator.ir.graph_delegate.GraphDelegate.add_edge

```{autodoc2-docstring} elasticai.creator.ir.graph_delegate.GraphDelegate.add_edge
```

````

````{py:method} add_node(node: elasticai.creator.ir.graph_delegate._T)
:canonical: elasticai.creator.ir.graph_delegate.GraphDelegate.add_node

```{autodoc2-docstring} elasticai.creator.ir.graph_delegate.GraphDelegate.add_node
```

````

````{py:method} iter_nodes() -> collections.abc.Iterator[elasticai.creator.ir.graph_delegate._T]
:canonical: elasticai.creator.ir.graph_delegate.GraphDelegate.iter_nodes

```{autodoc2-docstring} elasticai.creator.ir.graph_delegate.GraphDelegate.iter_nodes
```

````

````{py:method} iter_edges() -> collections.abc.Iterator[tuple[elasticai.creator.ir.graph_delegate._T, elasticai.creator.ir.graph_delegate._T]]
:canonical: elasticai.creator.ir.graph_delegate.GraphDelegate.iter_edges

```{autodoc2-docstring} elasticai.creator.ir.graph_delegate.GraphDelegate.iter_edges
```

````

````{py:method} get_edges() -> collections.abc.Iterator[tuple[elasticai.creator.ir.graph_delegate._T, elasticai.creator.ir.graph_delegate._T]]
:canonical: elasticai.creator.ir.graph_delegate.GraphDelegate.get_edges

```{autodoc2-docstring} elasticai.creator.ir.graph_delegate.GraphDelegate.get_edges
```

````

````{py:method} get_successors(node: elasticai.creator.ir.graph_delegate._T) -> collections.abc.Iterator[elasticai.creator.ir.graph_delegate._T]
:canonical: elasticai.creator.ir.graph_delegate.GraphDelegate.get_successors

```{autodoc2-docstring} elasticai.creator.ir.graph_delegate.GraphDelegate.get_successors
```

````

````{py:method} get_predecessors(node: elasticai.creator.ir.graph_delegate._T) -> collections.abc.Iterator[elasticai.creator.ir.graph_delegate._T]
:canonical: elasticai.creator.ir.graph_delegate.GraphDelegate.get_predecessors

```{autodoc2-docstring} elasticai.creator.ir.graph_delegate.GraphDelegate.get_predecessors
```

````

`````

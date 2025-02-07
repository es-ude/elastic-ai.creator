# {py:mod}`elasticai.creator.ir.graph_iterators`

```{py:module} elasticai.creator.ir.graph_iterators
```

```{autodoc2-docstring} elasticai.creator.ir.graph_iterators
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NodeNeighbourFn <elasticai.creator.ir.graph_iterators.NodeNeighbourFn>`
  - ```{autodoc2-docstring} elasticai.creator.ir.graph_iterators.NodeNeighbourFn
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`dfs_pre_order <elasticai.creator.ir.graph_iterators.dfs_pre_order>`
  - ```{autodoc2-docstring} elasticai.creator.ir.graph_iterators.dfs_pre_order
    :summary:
    ```
* - {py:obj}`bfs_iter_down <elasticai.creator.ir.graph_iterators.bfs_iter_down>`
  - ```{autodoc2-docstring} elasticai.creator.ir.graph_iterators.bfs_iter_down
    :summary:
    ```
* - {py:obj}`bfs_iter_up <elasticai.creator.ir.graph_iterators.bfs_iter_up>`
  - ```{autodoc2-docstring} elasticai.creator.ir.graph_iterators.bfs_iter_up
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HashableT <elasticai.creator.ir.graph_iterators.HashableT>`
  - ```{autodoc2-docstring} elasticai.creator.ir.graph_iterators.HashableT
    :summary:
    ```
````

### API

````{py:data} HashableT
:canonical: elasticai.creator.ir.graph_iterators.HashableT
:value: >
   'TypeVar(...)'

```{autodoc2-docstring} elasticai.creator.ir.graph_iterators.HashableT
```

````

`````{py:class} NodeNeighbourFn
:canonical: elasticai.creator.ir.graph_iterators.NodeNeighbourFn

Bases: {py:obj}`typing.Protocol`\[{py:obj}`elasticai.creator.ir.graph_iterators.HashableT`\]

```{autodoc2-docstring} elasticai.creator.ir.graph_iterators.NodeNeighbourFn
```

````{py:method} __call__(node: elasticai.creator.ir.graph_iterators.HashableT) -> collections.abc.Iterable[elasticai.creator.ir.graph_iterators.HashableT]
:canonical: elasticai.creator.ir.graph_iterators.NodeNeighbourFn.__call__

```{autodoc2-docstring} elasticai.creator.ir.graph_iterators.NodeNeighbourFn.__call__
```

````

`````

````{py:function} dfs_pre_order(successors: elasticai.creator.ir.graph_iterators.NodeNeighbourFn, start: elasticai.creator.ir.graph_iterators.HashableT) -> collections.abc.Iterator[elasticai.creator.ir.graph_iterators.HashableT]
:canonical: elasticai.creator.ir.graph_iterators.dfs_pre_order

```{autodoc2-docstring} elasticai.creator.ir.graph_iterators.dfs_pre_order
```
````

````{py:function} bfs_iter_down(successors: elasticai.creator.ir.graph_iterators.NodeNeighbourFn, predecessors: elasticai.creator.ir.graph_iterators.NodeNeighbourFn, start: elasticai.creator.ir.graph_iterators.HashableT) -> collections.abc.Iterator[elasticai.creator.ir.graph_iterators.HashableT]
:canonical: elasticai.creator.ir.graph_iterators.bfs_iter_down

```{autodoc2-docstring} elasticai.creator.ir.graph_iterators.bfs_iter_down
```
````

````{py:function} bfs_iter_up(predecessors: elasticai.creator.ir.graph_iterators.NodeNeighbourFn[elasticai.creator.ir.graph_iterators.HashableT], successors: elasticai.creator.ir.graph_iterators.NodeNeighbourFn[elasticai.creator.ir.graph_iterators.HashableT], start: elasticai.creator.ir.graph_iterators.HashableT) -> collections.abc.Iterator[elasticai.creator.ir.graph_iterators.HashableT]
:canonical: elasticai.creator.ir.graph_iterators.bfs_iter_up

```{autodoc2-docstring} elasticai.creator.ir.graph_iterators.bfs_iter_up
```
````

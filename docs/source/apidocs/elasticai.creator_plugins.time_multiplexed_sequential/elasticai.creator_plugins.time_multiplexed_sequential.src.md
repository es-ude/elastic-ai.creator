# {py:mod}`elasticai.creator_plugins.time_multiplexed_sequential.src`

```{py:module} elasticai.creator_plugins.time_multiplexed_sequential.src
```

```{autodoc2-docstring} elasticai.creator_plugins.time_multiplexed_sequential.src
:allowtitles:
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

elasticai.creator_plugins.time_multiplexed_sequential.src._imports
elasticai.creator_plugins.time_multiplexed_sequential.src.node_protocol
```

## Package Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_FilterNode <elasticai.creator_plugins.time_multiplexed_sequential.src._FilterNode>`
  -
* - {py:obj}`_Sequential <elasticai.creator_plugins.time_multiplexed_sequential.src._Sequential>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.time_multiplexed_sequential.src._Sequential
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_type_handler_fn <elasticai.creator_plugins.time_multiplexed_sequential.src._type_handler_fn>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.time_multiplexed_sequential.src._type_handler_fn
    :summary:
    ```
* - {py:obj}`_iterable_type_handler_fn <elasticai.creator_plugins.time_multiplexed_sequential.src._iterable_type_handler_fn>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.time_multiplexed_sequential.src._iterable_type_handler_fn
    :summary:
    ```
* - {py:obj}`append_counter_suffix_before_construction <elasticai.creator_plugins.time_multiplexed_sequential.src.append_counter_suffix_before_construction>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.time_multiplexed_sequential.src.append_counter_suffix_before_construction
    :summary:
    ```
* - {py:obj}`sequential <elasticai.creator_plugins.time_multiplexed_sequential.src.sequential>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.time_multiplexed_sequential.src.sequential
    :summary:
    ```
* - {py:obj}`network <elasticai.creator_plugins.time_multiplexed_sequential.src.network>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.time_multiplexed_sequential.src.network
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`P <elasticai.creator_plugins.time_multiplexed_sequential.src.P>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.time_multiplexed_sequential.src.P
    :summary:
    ```
* - {py:obj}`_type_handler <elasticai.creator_plugins.time_multiplexed_sequential.src._type_handler>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.time_multiplexed_sequential.src._type_handler
    :summary:
    ```
* - {py:obj}`_iterable_type_handler <elasticai.creator_plugins.time_multiplexed_sequential.src._iterable_type_handler>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.time_multiplexed_sequential.src._iterable_type_handler
    :summary:
    ```
````

### API

````{py:data} P
:canonical: elasticai.creator_plugins.time_multiplexed_sequential.src.P
:value: >
   'ParamSpec(...)'

```{autodoc2-docstring} elasticai.creator_plugins.time_multiplexed_sequential.src.P
```

````

````{py:function} _type_handler_fn(name: str, fn: collections.abc.Callable[[elasticai.creator.ir2vhdl.Implementation], elasticai.creator.ir2vhdl.Implementation]) -> elasticai.creator.plugin.PluginSymbolFn[elasticai.creator.ir2vhdl.LoweringPass, [elasticai.creator.ir2vhdl.Implementation], elasticai.creator.ir2vhdl.Implementation]
:canonical: elasticai.creator_plugins.time_multiplexed_sequential.src._type_handler_fn

```{autodoc2-docstring} elasticai.creator_plugins.time_multiplexed_sequential.src._type_handler_fn
```
````

````{py:function} _iterable_type_handler_fn(name: str, fn: collections.abc.Callable[[elasticai.creator.ir2vhdl.Implementation], collections.abc.Iterable[elasticai.creator.ir2vhdl.Implementation]]) -> elasticai.creator.plugin.PluginSymbolFn[elasticai.creator.ir2vhdl.LoweringPass, [elasticai.creator.ir2vhdl.Implementation], collections.abc.Iterable[elasticai.creator.ir2vhdl.Implementation]]
:canonical: elasticai.creator_plugins.time_multiplexed_sequential.src._iterable_type_handler_fn

```{autodoc2-docstring} elasticai.creator_plugins.time_multiplexed_sequential.src._iterable_type_handler_fn
```
````

````{py:data} _type_handler
:canonical: elasticai.creator_plugins.time_multiplexed_sequential.src._type_handler
:value: >
   'FunctionDecorator(...)'

```{autodoc2-docstring} elasticai.creator_plugins.time_multiplexed_sequential.src._type_handler
```

````

````{py:data} _iterable_type_handler
:canonical: elasticai.creator_plugins.time_multiplexed_sequential.src._iterable_type_handler
:value: >
   'FunctionDecorator(...)'

```{autodoc2-docstring} elasticai.creator_plugins.time_multiplexed_sequential.src._iterable_type_handler
```

````

`````{py:class} _FilterNode(data: dict[str, elasticai.creator.ir.attribute.Attribute])
:canonical: elasticai.creator_plugins.time_multiplexed_sequential.src._FilterNode

Bases: {py:obj}`elasticai.creator.ir2vhdl.VhdlNode`

````{py:attribute} params
:canonical: elasticai.creator_plugins.time_multiplexed_sequential.src._FilterNode.params
:type: elasticai.creator.ir.RequiredField[dict, elasticai.creator_plugins.grouped_filter.FilterParameters]
:value: >
   'RequiredField(...)'

```{autodoc2-docstring} elasticai.creator_plugins.time_multiplexed_sequential.src._FilterNode.params
```

````

`````

````{py:function} append_counter_suffix_before_construction(fn: collections.abc.Callable[elasticai.creator_plugins.time_multiplexed_sequential.src.P, elasticai.creator.ir2vhdl.VhdlNode]) -> collections.abc.Callable[elasticai.creator_plugins.time_multiplexed_sequential.src.P, elasticai.creator.ir2vhdl.VhdlNode]
:canonical: elasticai.creator_plugins.time_multiplexed_sequential.src.append_counter_suffix_before_construction

```{autodoc2-docstring} elasticai.creator_plugins.time_multiplexed_sequential.src.append_counter_suffix_before_construction
```
````

`````{py:class} _Sequential(name: str)
:canonical: elasticai.creator_plugins.time_multiplexed_sequential.src._Sequential

```{autodoc2-docstring} elasticai.creator_plugins.time_multiplexed_sequential.src._Sequential
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator_plugins.time_multiplexed_sequential.src._Sequential.__init__
```

````{py:method} add_input(shape: elasticai.creator.ir2vhdl.Shape)
:canonical: elasticai.creator_plugins.time_multiplexed_sequential.src._Sequential.add_input

```{autodoc2-docstring} elasticai.creator_plugins.time_multiplexed_sequential.src._Sequential.add_input
```

````

````{py:method} filter(n: elasticai.creator.ir2vhdl.VhdlNode)
:canonical: elasticai.creator_plugins.time_multiplexed_sequential.src._Sequential.filter

```{autodoc2-docstring} elasticai.creator_plugins.time_multiplexed_sequential.src._Sequential.filter
```

````

````{py:method} _append_node(name: str, output_shape: elasticai.creator.ir2vhdl.Shape, type: str, implementation: str, node_fn, attributes=None)
:canonical: elasticai.creator_plugins.time_multiplexed_sequential.src._Sequential._append_node

```{autodoc2-docstring} elasticai.creator_plugins.time_multiplexed_sequential.src._Sequential._append_node
```

````

````{py:method} _append_static(name: str, implementation: str, output_shape: elasticai.creator.ir2vhdl.Shape, **kwargs)
:canonical: elasticai.creator_plugins.time_multiplexed_sequential.src._Sequential._append_static

```{autodoc2-docstring} elasticai.creator_plugins.time_multiplexed_sequential.src._Sequential._append_static
```

````

````{py:method} shift_register(name: str, output_shape: elasticai.creator.ir2vhdl.Shape)
:canonical: elasticai.creator_plugins.time_multiplexed_sequential.src._Sequential.shift_register

```{autodoc2-docstring} elasticai.creator_plugins.time_multiplexed_sequential.src._Sequential.shift_register
```

````

````{py:method} strided_shift_register(output_shape: tuple[int, int], stride: int)
:canonical: elasticai.creator_plugins.time_multiplexed_sequential.src._Sequential.strided_shift_register

```{autodoc2-docstring} elasticai.creator_plugins.time_multiplexed_sequential.src._Sequential.strided_shift_register
```

````

````{py:method} input(impl: elasticai.creator.ir2vhdl.Implementation) -> None
:canonical: elasticai.creator_plugins.time_multiplexed_sequential.src._Sequential.input

```{autodoc2-docstring} elasticai.creator_plugins.time_multiplexed_sequential.src._Sequential.input
```

````

````{py:method} _determine_required_input_shape(impl: elasticai.creator.ir2vhdl.Implementation)
:canonical: elasticai.creator_plugins.time_multiplexed_sequential.src._Sequential._determine_required_input_shape

```{autodoc2-docstring} elasticai.creator_plugins.time_multiplexed_sequential.src._Sequential._determine_required_input_shape
```

````

````{py:method} set_runtime_input_shape(s: elasticai.creator.ir2vhdl.Shape) -> None
:canonical: elasticai.creator_plugins.time_multiplexed_sequential.src._Sequential.set_runtime_input_shape

```{autodoc2-docstring} elasticai.creator_plugins.time_multiplexed_sequential.src._Sequential.set_runtime_input_shape
```

````

````{py:method} set_runtime_output_shape(s: elasticai.creator.ir2vhdl.Shape) -> None
:canonical: elasticai.creator_plugins.time_multiplexed_sequential.src._Sequential.set_runtime_output_shape

```{autodoc2-docstring} elasticai.creator_plugins.time_multiplexed_sequential.src._Sequential.set_runtime_output_shape
```

````

````{py:method} get_impl() -> elasticai.creator.ir2vhdl.Implementation
:canonical: elasticai.creator_plugins.time_multiplexed_sequential.src._Sequential.get_impl

```{autodoc2-docstring} elasticai.creator_plugins.time_multiplexed_sequential.src._Sequential.get_impl
```

````

````{py:method} finish()
:canonical: elasticai.creator_plugins.time_multiplexed_sequential.src._Sequential.finish

```{autodoc2-docstring} elasticai.creator_plugins.time_multiplexed_sequential.src._Sequential.finish
```

````

`````

````{py:function} sequential(impl: elasticai.creator.ir2vhdl.Implementation) -> elasticai.creator.ir2vhdl.Implementation
:canonical: elasticai.creator_plugins.time_multiplexed_sequential.src.sequential

```{autodoc2-docstring} elasticai.creator_plugins.time_multiplexed_sequential.src.sequential
```
````

````{py:function} network(impl: elasticai.creator.ir2vhdl.Implementation) -> collections.abc.Iterable[elasticai.creator.ir2vhdl.Implementation]
:canonical: elasticai.creator_plugins.time_multiplexed_sequential.src.network

```{autodoc2-docstring} elasticai.creator_plugins.time_multiplexed_sequential.src.network
```
````

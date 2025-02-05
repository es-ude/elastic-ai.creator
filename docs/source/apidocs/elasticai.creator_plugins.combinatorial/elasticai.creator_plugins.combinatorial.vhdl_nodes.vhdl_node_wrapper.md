# {py:mod}`elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper`

```{py:module} elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper
```

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Definable <elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper.Definable>`
  -
* - {py:obj}`Node <elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper.Node>`
  -
* - {py:obj}`VHDLNodeWrapper <elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper.VHDLNodeWrapper>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper.VHDLNodeWrapper
    :summary:
    ```
````

### API

`````{py:class} Definable
:canonical: elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper.Definable

Bases: {py:obj}`typing.Protocol`

````{py:property} name
:canonical: elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper.Definable.name
:abstractmethod:
:type: str

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper.Definable.name
```

````

````{py:method} define() -> typing.Iterator[str]
:canonical: elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper.Definable.define
:abstractmethod:

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper.Definable.define
```

````

`````

`````{py:class} Node
:canonical: elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper.Node

Bases: {py:obj}`typing.Protocol`

````{py:attribute} implementation
:canonical: elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper.Node.implementation
:type: str
:value: >
   None

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper.Node.implementation
```

````

````{py:attribute} name
:canonical: elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper.Node.name
:type: str
:value: >
   None

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper.Node.name
```

````

````{py:attribute} type
:canonical: elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper.Node.type
:type: str
:value: >
   None

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper.Node.type
```

````

````{py:attribute} input_shape
:canonical: elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper.Node.input_shape
:type: tuple[int, int]
:value: >
   None

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper.Node.input_shape
```

````

````{py:attribute} output_shape
:canonical: elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper.Node.output_shape
:type: tuple[int, int]
:value: >
   None

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper.Node.output_shape
```

````

````{py:attribute} attributes
:canonical: elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper.Node.attributes
:type: dict[str, typing.Any]
:value: >
   None

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper.Node.attributes
```

````

`````

`````{py:class} VHDLNodeWrapper(node: elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper.Node, generic_map: dict[str, str], port_map: dict[str, elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper.Definable])
:canonical: elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper.VHDLNodeWrapper

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper.VHDLNodeWrapper
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper.VHDLNodeWrapper.__init__
```

````{py:method} add_signal_with_suffix(signal: elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper.Definable, prefix: str | None = None) -> None
:canonical: elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper.VHDLNodeWrapper.add_signal_with_suffix

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper.VHDLNodeWrapper.add_signal_with_suffix
```

````

````{py:property} name
:canonical: elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper.VHDLNodeWrapper.name
:type: str

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper.VHDLNodeWrapper.name
```

````

````{py:property} implementation
:canonical: elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper.VHDLNodeWrapper.implementation
:type: str

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper.VHDLNodeWrapper.implementation
```

````

````{py:method} define_signals() -> typing.Iterator[str]
:canonical: elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper.VHDLNodeWrapper.define_signals

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper.VHDLNodeWrapper.define_signals
```

````

````{py:method} generate_entity() -> typing.Iterator[str]
:canonical: elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper.VHDLNodeWrapper.generate_entity

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper.VHDLNodeWrapper.generate_entity
```

````

````{py:method} instantiate() -> typing.Iterator[str]
:canonical: elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper.VHDLNodeWrapper.instantiate

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.vhdl_node_wrapper.VHDLNodeWrapper.instantiate
```

````

`````

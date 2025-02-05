# {py:mod}`elasticai.creator_plugins.combinatorial.vhdl_nodes.test_vhdl_nodes`

```{py:module} elasticai.creator_plugins.combinatorial.vhdl_nodes.test_vhdl_nodes
```

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.test_vhdl_nodes
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TestStridingShiftRegister <elasticai.creator_plugins.combinatorial.vhdl_nodes.test_vhdl_nodes.TestStridingShiftRegister>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.test_vhdl_nodes.TestStridingShiftRegister
    :summary:
    ```
* - {py:obj}`TestShiftRegister <elasticai.creator_plugins.combinatorial.vhdl_nodes.test_vhdl_nodes.TestShiftRegister>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.test_vhdl_nodes.TestShiftRegister
    :summary:
    ```
* - {py:obj}`TestSlidingWindow <elasticai.creator_plugins.combinatorial.vhdl_nodes.test_vhdl_nodes.TestSlidingWindow>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.test_vhdl_nodes.TestSlidingWindow
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`node <elasticai.creator_plugins.combinatorial.vhdl_nodes.test_vhdl_nodes.node>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.test_vhdl_nodes.node
    :summary:
    ```
* - {py:obj}`new_node <elasticai.creator_plugins.combinatorial.vhdl_nodes.test_vhdl_nodes.new_node>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.test_vhdl_nodes.new_node
    :summary:
    ```
````

### API

````{py:function} node(raw_node)
:canonical: elasticai.creator_plugins.combinatorial.vhdl_nodes.test_vhdl_nodes.node

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.test_vhdl_nodes.node
```
````

````{py:function} new_node(name: str, type: str, implementation: str, input_shape: elasticai.creator.ir2vhdl.Shape, output_shape: elasticai.creator.ir2vhdl.Shape, attributes: dict | None = None) -> elasticai.creator.ir2vhdl.VhdlNode
:canonical: elasticai.creator_plugins.combinatorial.vhdl_nodes.test_vhdl_nodes.new_node

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.test_vhdl_nodes.new_node
```
````

`````{py:class} TestStridingShiftRegister
:canonical: elasticai.creator_plugins.combinatorial.vhdl_nodes.test_vhdl_nodes.TestStridingShiftRegister

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.test_vhdl_nodes.TestStridingShiftRegister
```

````{py:method} raw_node()
:canonical: elasticai.creator_plugins.combinatorial.vhdl_nodes.test_vhdl_nodes.TestStridingShiftRegister.raw_node

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.test_vhdl_nodes.TestStridingShiftRegister.raw_node
```

````

````{py:method} test_can_instantiate(node, raw_node)
:canonical: elasticai.creator_plugins.combinatorial.vhdl_nodes.test_vhdl_nodes.TestStridingShiftRegister.test_can_instantiate

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.test_vhdl_nodes.TestStridingShiftRegister.test_can_instantiate
```

````

````{py:method} test_can_define_signals(node, raw_node)
:canonical: elasticai.creator_plugins.combinatorial.vhdl_nodes.test_vhdl_nodes.TestStridingShiftRegister.test_can_define_signals

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.test_vhdl_nodes.TestStridingShiftRegister.test_can_define_signals
```

````

`````

`````{py:class} TestShiftRegister
:canonical: elasticai.creator_plugins.combinatorial.vhdl_nodes.test_vhdl_nodes.TestShiftRegister

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.test_vhdl_nodes.TestShiftRegister
```

````{py:method} raw_node()
:canonical: elasticai.creator_plugins.combinatorial.vhdl_nodes.test_vhdl_nodes.TestShiftRegister.raw_node

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.test_vhdl_nodes.TestShiftRegister.raw_node
```

````

````{py:method} test_can_instantiate(node, raw_node)
:canonical: elasticai.creator_plugins.combinatorial.vhdl_nodes.test_vhdl_nodes.TestShiftRegister.test_can_instantiate

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.test_vhdl_nodes.TestShiftRegister.test_can_instantiate
```

````

````{py:method} test_can_define_signals(node)
:canonical: elasticai.creator_plugins.combinatorial.vhdl_nodes.test_vhdl_nodes.TestShiftRegister.test_can_define_signals

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.test_vhdl_nodes.TestShiftRegister.test_can_define_signals
```

````

`````

`````{py:class} TestSlidingWindow
:canonical: elasticai.creator_plugins.combinatorial.vhdl_nodes.test_vhdl_nodes.TestSlidingWindow

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.test_vhdl_nodes.TestSlidingWindow
```

````{py:method} raw_node()
:canonical: elasticai.creator_plugins.combinatorial.vhdl_nodes.test_vhdl_nodes.TestSlidingWindow.raw_node

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.test_vhdl_nodes.TestSlidingWindow.raw_node
```

````

````{py:method} test_can_instantiate(node)
:canonical: elasticai.creator_plugins.combinatorial.vhdl_nodes.test_vhdl_nodes.TestSlidingWindow.test_can_instantiate

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.test_vhdl_nodes.TestSlidingWindow.test_can_instantiate
```

````

````{py:method} test_can_define_signals(node)
:canonical: elasticai.creator_plugins.combinatorial.vhdl_nodes.test_vhdl_nodes.TestSlidingWindow.test_can_define_signals

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.test_vhdl_nodes.TestSlidingWindow.test_can_define_signals
```

````

`````

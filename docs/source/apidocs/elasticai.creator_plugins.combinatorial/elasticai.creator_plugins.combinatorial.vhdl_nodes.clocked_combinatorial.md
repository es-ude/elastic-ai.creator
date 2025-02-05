# {py:mod}`elasticai.creator_plugins.combinatorial.vhdl_nodes.clocked_combinatorial`

```{py:module} elasticai.creator_plugins.combinatorial.vhdl_nodes.clocked_combinatorial
```

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.clocked_combinatorial
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ClockedInstance <elasticai.creator_plugins.combinatorial.vhdl_nodes.clocked_combinatorial.ClockedInstance>`
  -
* - {py:obj}`_ClockedCombinatorial <elasticai.creator_plugins.combinatorial.vhdl_nodes.clocked_combinatorial._ClockedCombinatorial>`
  -
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`clocked_combinatorial <elasticai.creator_plugins.combinatorial.vhdl_nodes.clocked_combinatorial.clocked_combinatorial>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.clocked_combinatorial.clocked_combinatorial
    :summary:
    ```
````

### API

````{py:function} clocked_combinatorial(node)
:canonical: elasticai.creator_plugins.combinatorial.vhdl_nodes.clocked_combinatorial.clocked_combinatorial

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.clocked_combinatorial.clocked_combinatorial
```
````

`````{py:class} ClockedInstance(node: elasticai.creator.ir2vhdl.VhdlNode, input_width: int, output_width: int, generic_map: dict[str, typing.Any] | typing.Callable[[], dict[str, typing.Any]] = lambda: {})
:canonical: elasticai.creator_plugins.combinatorial.vhdl_nodes.clocked_combinatorial.ClockedInstance

Bases: {py:obj}`elasticai.creator.ir2vhdl.Instance`

````{py:attribute} _logic_signals_with_default_suffix
:canonical: elasticai.creator_plugins.combinatorial.vhdl_nodes.clocked_combinatorial.ClockedInstance._logic_signals_with_default_suffix
:type: tuple[str, ...]
:value: >
   'tuple(...)'

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.clocked_combinatorial.ClockedInstance._logic_signals_with_default_suffix
```

````

````{py:attribute} _vector_signals_with_default_suffix
:canonical: elasticai.creator_plugins.combinatorial.vhdl_nodes.clocked_combinatorial.ClockedInstance._vector_signals_with_default_suffix
:type: tuple[tuple[str, int], ...]
:value: >
   'tuple(...)'

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.clocked_combinatorial.ClockedInstance._vector_signals_with_default_suffix
```

````

`````

`````{py:class} _ClockedCombinatorial(node: elasticai.creator.ir2vhdl.VhdlNode, input_width: int, output_width: int, generic_map: dict[str, typing.Any] | typing.Callable[[], dict[str, typing.Any]] = lambda: {})
:canonical: elasticai.creator_plugins.combinatorial.vhdl_nodes.clocked_combinatorial._ClockedCombinatorial

Bases: {py:obj}`elasticai.creator_plugins.combinatorial.vhdl_nodes.clocked_combinatorial.ClockedInstance`

````{py:attribute} _logic_signals_with_default_suffix
:canonical: elasticai.creator_plugins.combinatorial.vhdl_nodes.clocked_combinatorial._ClockedCombinatorial._logic_signals_with_default_suffix
:value: >
   ('valid_in', 'valid_out')

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.vhdl_nodes.clocked_combinatorial._ClockedCombinatorial._logic_signals_with_default_suffix
```

````

`````

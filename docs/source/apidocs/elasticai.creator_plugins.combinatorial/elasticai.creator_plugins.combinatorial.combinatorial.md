# {py:mod}`elasticai.creator_plugins.combinatorial.combinatorial`

```{py:module} elasticai.creator_plugins.combinatorial.combinatorial
```

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.combinatorial
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`wrap_node <elasticai.creator_plugins.combinatorial.combinatorial.wrap_node>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.combinatorial.wrap_node
    :summary:
    ```
* - {py:obj}`build_declarations_for_combinatorial <elasticai.creator_plugins.combinatorial.combinatorial.build_declarations_for_combinatorial>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.combinatorial.build_declarations_for_combinatorial
    :summary:
    ```
* - {py:obj}`build_instantiations_for_combinatorial <elasticai.creator_plugins.combinatorial.combinatorial.build_instantiations_for_combinatorial>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.combinatorial.build_instantiations_for_combinatorial
    :summary:
    ```
* - {py:obj}`build_data_signal_connections_for_combinatorial <elasticai.creator_plugins.combinatorial.combinatorial.build_data_signal_connections_for_combinatorial>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.combinatorial.build_data_signal_connections_for_combinatorial
    :summary:
    ```
* - {py:obj}`wrap_in_architecture <elasticai.creator_plugins.combinatorial.combinatorial.wrap_in_architecture>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.combinatorial.wrap_in_architecture
    :summary:
    ```
````

### API

````{py:function} wrap_node(n)
:canonical: elasticai.creator_plugins.combinatorial.combinatorial.wrap_node

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.combinatorial.wrap_node
```
````

````{py:function} build_declarations_for_combinatorial(impl: elasticai.creator.ir2vhdl.Implementation) -> collections.abc.Iterator[str]
:canonical: elasticai.creator_plugins.combinatorial.combinatorial.build_declarations_for_combinatorial

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.combinatorial.build_declarations_for_combinatorial
```
````

````{py:function} build_instantiations_for_combinatorial(impl: elasticai.creator.ir2vhdl.Implementation) -> collections.abc.Iterator[str]
:canonical: elasticai.creator_plugins.combinatorial.combinatorial.build_instantiations_for_combinatorial

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.combinatorial.build_instantiations_for_combinatorial
```
````

````{py:function} build_data_signal_connections_for_combinatorial(impl: elasticai.creator.ir2vhdl.Implementation) -> collections.abc.Iterator[str]
:canonical: elasticai.creator_plugins.combinatorial.combinatorial.build_data_signal_connections_for_combinatorial

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.combinatorial.build_data_signal_connections_for_combinatorial
```
````

````{py:function} wrap_in_architecture(name, declarations, definitions)
:canonical: elasticai.creator_plugins.combinatorial.combinatorial.wrap_in_architecture

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.combinatorial.wrap_in_architecture
```
````

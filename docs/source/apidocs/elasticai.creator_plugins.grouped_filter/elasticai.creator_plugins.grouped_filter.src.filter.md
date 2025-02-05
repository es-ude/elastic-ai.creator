# {py:mod}`elasticai.creator_plugins.grouped_filter.src.filter`

```{py:module} elasticai.creator_plugins.grouped_filter.src.filter
```

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.filter
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_type_handler_fn <elasticai.creator_plugins.grouped_filter.src.filter._type_handler_fn>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.filter._type_handler_fn
    :summary:
    ```
* - {py:obj}`append_counter_suffix_before_construction <elasticai.creator_plugins.grouped_filter.src.filter.append_counter_suffix_before_construction>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.filter.append_counter_suffix_before_construction
    :summary:
    ```
* - {py:obj}`grouped_filter <elasticai.creator_plugins.grouped_filter.src.filter.grouped_filter>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.filter.grouped_filter
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`P <elasticai.creator_plugins.grouped_filter.src.filter.P>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.filter.P
    :summary:
    ```
* - {py:obj}`_type_handler <elasticai.creator_plugins.grouped_filter.src.filter._type_handler>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.filter._type_handler
    :summary:
    ```
````

### API

````{py:data} P
:canonical: elasticai.creator_plugins.grouped_filter.src.filter.P
:value: >
   'ParamSpec(...)'

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.filter.P
```

````

````{py:function} _type_handler_fn(name: str, fn: collections.abc.Callable[[elasticai.creator.ir2vhdl.Implementation], elasticai.creator.ir2vhdl.Implementation]) -> elasticai.creator.plugin.PluginSymbol
:canonical: elasticai.creator_plugins.grouped_filter.src.filter._type_handler_fn

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.filter._type_handler_fn
```
````

````{py:data} _type_handler
:canonical: elasticai.creator_plugins.grouped_filter.src.filter._type_handler
:value: >
   'FunctionDecorator(...)'

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.filter._type_handler
```

````

````{py:function} append_counter_suffix_before_construction(fn: collections.abc.Callable[elasticai.creator_plugins.grouped_filter.src.filter.P, elasticai.creator.ir2vhdl.VhdlNode]) -> collections.abc.Callable[elasticai.creator_plugins.grouped_filter.src.filter.P, elasticai.creator.ir2vhdl.VhdlNode]
:canonical: elasticai.creator_plugins.grouped_filter.src.filter.append_counter_suffix_before_construction

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.filter.append_counter_suffix_before_construction
```
````

````{py:function} grouped_filter(impl: elasticai.creator.ir2vhdl.Implementation) -> elasticai.creator.ir2vhdl.Implementation
:canonical: elasticai.creator_plugins.grouped_filter.src.filter.grouped_filter

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.filter.grouped_filter
```
````

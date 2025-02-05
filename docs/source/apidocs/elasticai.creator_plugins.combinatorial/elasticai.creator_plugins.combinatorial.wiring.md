# {py:mod}`elasticai.creator_plugins.combinatorial.wiring`

```{py:module} elasticai.creator_plugins.combinatorial.wiring
```

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.wiring
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Shape <elasticai.creator_plugins.combinatorial.wiring.Shape>`
  -
* - {py:obj}`Node <elasticai.creator_plugins.combinatorial.wiring.Node>`
  -
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_for_each_produce_multiple_lines <elasticai.creator_plugins.combinatorial.wiring._for_each_produce_multiple_lines>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.wiring._for_each_produce_multiple_lines
    :summary:
    ```
* - {py:obj}`_for_each_produce_single_line <elasticai.creator_plugins.combinatorial.wiring._for_each_produce_single_line>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.wiring._for_each_produce_single_line
    :summary:
    ```
* - {py:obj}`connect_data_signals <elasticai.creator_plugins.combinatorial.wiring.connect_data_signals>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.wiring.connect_data_signals
    :summary:
    ```
* - {py:obj}`_define_signal_vector <elasticai.creator_plugins.combinatorial.wiring._define_signal_vector>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.wiring._define_signal_vector
    :summary:
    ```
* - {py:obj}`define_input_data_signals <elasticai.creator_plugins.combinatorial.wiring.define_input_data_signals>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.wiring.define_input_data_signals
    :summary:
    ```
* - {py:obj}`define_output_data_signals <elasticai.creator_plugins.combinatorial.wiring.define_output_data_signals>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.wiring.define_output_data_signals
    :summary:
    ```
* - {py:obj}`instantiate_bufferless <elasticai.creator_plugins.combinatorial.wiring.instantiate_bufferless>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.wiring.instantiate_bufferless
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_T <elasticai.creator_plugins.combinatorial.wiring._T>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.wiring._T
    :summary:
    ```
````

### API

````{py:data} _T
:canonical: elasticai.creator_plugins.combinatorial.wiring._T
:value: >
   'TypeVar(...)'

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.wiring._T
```

````

`````{py:class} Shape
:canonical: elasticai.creator_plugins.combinatorial.wiring.Shape

Bases: {py:obj}`typing.Protocol`

````{py:attribute} width
:canonical: elasticai.creator_plugins.combinatorial.wiring.Shape.width
:type: int
:value: >
   None

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.wiring.Shape.width
```

````

````{py:method} size() -> int
:canonical: elasticai.creator_plugins.combinatorial.wiring.Shape.size

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.wiring.Shape.size
```

````

`````

`````{py:class} Node
:canonical: elasticai.creator_plugins.combinatorial.wiring.Node

Bases: {py:obj}`typing.Protocol`

````{py:attribute} name
:canonical: elasticai.creator_plugins.combinatorial.wiring.Node.name
:type: str
:value: >
   None

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.wiring.Node.name
```

````

````{py:attribute} implementation
:canonical: elasticai.creator_plugins.combinatorial.wiring.Node.implementation
:type: str
:value: >
   None

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.wiring.Node.implementation
```

````

````{py:attribute} input_shape
:canonical: elasticai.creator_plugins.combinatorial.wiring.Node.input_shape
:type: elasticai.creator_plugins.combinatorial.wiring.Shape
:value: >
   None

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.wiring.Node.input_shape
```

````

````{py:attribute} output_shape
:canonical: elasticai.creator_plugins.combinatorial.wiring.Node.output_shape
:type: elasticai.creator_plugins.combinatorial.wiring.Shape
:value: >
   None

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.wiring.Node.output_shape
```

````

`````

````{py:function} _for_each_produce_multiple_lines(fn: collections.abc.Callable[[elasticai.creator_plugins.combinatorial.wiring._T], collections.abc.Iterable[str]]) -> collections.abc.Callable[[collections.abc.Iterable[elasticai.creator_plugins.combinatorial.wiring._T]], list[str]]
:canonical: elasticai.creator_plugins.combinatorial.wiring._for_each_produce_multiple_lines

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.wiring._for_each_produce_multiple_lines
```
````

````{py:function} _for_each_produce_single_line(fn: collections.abc.Callable[[elasticai.creator_plugins.combinatorial.wiring._T], str]) -> collections.abc.Callable[[collections.abc.Iterable[elasticai.creator_plugins.combinatorial.wiring._T]], list[str]]
:canonical: elasticai.creator_plugins.combinatorial.wiring._for_each_produce_single_line

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.wiring._for_each_produce_single_line
```
````

````{py:function} connect_data_signals(edge: tuple[str, str, collections.abc.Sequence[tuple[int, int]] | tuple[str, str]])
:canonical: elasticai.creator_plugins.combinatorial.wiring.connect_data_signals

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.wiring.connect_data_signals
```
````

````{py:function} _define_signal_vector(name, length)
:canonical: elasticai.creator_plugins.combinatorial.wiring._define_signal_vector

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.wiring._define_signal_vector
```
````

````{py:function} define_input_data_signals(instance)
:canonical: elasticai.creator_plugins.combinatorial.wiring.define_input_data_signals

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.wiring.define_input_data_signals
```
````

````{py:function} define_output_data_signals(instance: elasticai.creator_plugins.combinatorial.wiring.Node)
:canonical: elasticai.creator_plugins.combinatorial.wiring.define_output_data_signals

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.wiring.define_output_data_signals
```
````

````{py:function} instantiate_bufferless(instance: elasticai.creator_plugins.combinatorial.wiring.Node)
:canonical: elasticai.creator_plugins.combinatorial.wiring.instantiate_bufferless

```{autodoc2-docstring} elasticai.creator_plugins.combinatorial.wiring.instantiate_bufferless
```
````

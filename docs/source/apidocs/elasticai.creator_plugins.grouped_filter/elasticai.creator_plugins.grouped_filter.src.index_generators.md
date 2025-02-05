# {py:mod}`elasticai.creator_plugins.grouped_filter.src.index_generators`

```{py:module} elasticai.creator_plugins.grouped_filter.src.index_generators
```

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.index_generators
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GroupedFilterIndexGenerator <elasticai.creator_plugins.grouped_filter.src.index_generators.GroupedFilterIndexGenerator>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.index_generators.GroupedFilterIndexGenerator
    :summary:
    ```
* - {py:obj}`Step <elasticai.creator_plugins.grouped_filter.src.index_generators.Step>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.index_generators.Step
    :summary:
    ```
* - {py:obj}`Group <elasticai.creator_plugins.grouped_filter.src.index_generators.Group>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.index_generators.Group
    :summary:
    ```
* - {py:obj}`IndicesForOperation <elasticai.creator_plugins.grouped_filter.src.index_generators.IndicesForOperation>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.index_generators.IndicesForOperation
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`channelwise_interleaved <elasticai.creator_plugins.grouped_filter.src.index_generators.channelwise_interleaved>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.index_generators.channelwise_interleaved
    :summary:
    ```
* - {py:obj}`groupwise <elasticai.creator_plugins.grouped_filter.src.index_generators.groupwise>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.index_generators.groupwise
    :summary:
    ```
* - {py:obj}`generate_deinterleaved_indices <elasticai.creator_plugins.grouped_filter.src.index_generators.generate_deinterleaved_indices>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.index_generators.generate_deinterleaved_indices
    :summary:
    ```
* - {py:obj}`sliding_window <elasticai.creator_plugins.grouped_filter.src.index_generators.sliding_window>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.index_generators.sliding_window
    :summary:
    ```
* - {py:obj}`unrolled_grouped_convolution <elasticai.creator_plugins.grouped_filter.src.index_generators.unrolled_grouped_convolution>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.index_generators.unrolled_grouped_convolution
    :summary:
    ```
````

### API

````{py:function} channelwise_interleaved(length, channels)
:canonical: elasticai.creator_plugins.grouped_filter.src.index_generators.channelwise_interleaved

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.index_generators.channelwise_interleaved
```
````

````{py:function} groupwise(length, channels, groups)
:canonical: elasticai.creator_plugins.grouped_filter.src.index_generators.groupwise

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.index_generators.groupwise
```
````

````{py:function} generate_deinterleaved_indices(length, channels, channel_size)
:canonical: elasticai.creator_plugins.grouped_filter.src.index_generators.generate_deinterleaved_indices

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.index_generators.generate_deinterleaved_indices
```
````

````{py:function} sliding_window(steps, size, stride)
:canonical: elasticai.creator_plugins.grouped_filter.src.index_generators.sliding_window

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.index_generators.sliding_window
```
````

`````{py:class} GroupedFilterIndexGenerator(params: elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters)
:canonical: elasticai.creator_plugins.grouped_filter.src.index_generators.GroupedFilterIndexGenerator

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.index_generators.GroupedFilterIndexGenerator
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.index_generators.GroupedFilterIndexGenerator.__init__
```

````{py:method} groups() -> typing.Iterator[elasticai.creator_plugins.grouped_filter.src.index_generators.Group]
:canonical: elasticai.creator_plugins.grouped_filter.src.index_generators.GroupedFilterIndexGenerator.groups

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.index_generators.GroupedFilterIndexGenerator.groups
```

````

````{py:method} steps() -> typing.Iterator[elasticai.creator_plugins.grouped_filter.src.index_generators.Step]
:canonical: elasticai.creator_plugins.grouped_filter.src.index_generators.GroupedFilterIndexGenerator.steps

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.index_generators.GroupedFilterIndexGenerator.steps
```

````

````{py:method} as_tuple_by_steps()
:canonical: elasticai.creator_plugins.grouped_filter.src.index_generators.GroupedFilterIndexGenerator.as_tuple_by_steps

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.index_generators.GroupedFilterIndexGenerator.as_tuple_by_steps
```

````

````{py:method} as_tuple_by_groups()
:canonical: elasticai.creator_plugins.grouped_filter.src.index_generators.GroupedFilterIndexGenerator.as_tuple_by_groups

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.index_generators.GroupedFilterIndexGenerator.as_tuple_by_groups
```

````

`````

`````{py:class} Step(params: elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters, step_id: int)
:canonical: elasticai.creator_plugins.grouped_filter.src.index_generators.Step

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.index_generators.Step
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.index_generators.Step.__init__
```

````{py:method} groups() -> typing.Iterator[elasticai.creator_plugins.grouped_filter.src.index_generators.IndicesForOperation]
:canonical: elasticai.creator_plugins.grouped_filter.src.index_generators.Step.groups

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.index_generators.Step.groups
```

````

`````

`````{py:class} Group(params: elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters, group_id: int)
:canonical: elasticai.creator_plugins.grouped_filter.src.index_generators.Group

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.index_generators.Group
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.index_generators.Group.__init__
```

````{py:method} steps() -> typing.Iterator[elasticai.creator_plugins.grouped_filter.src.index_generators.IndicesForOperation]
:canonical: elasticai.creator_plugins.grouped_filter.src.index_generators.Group.steps

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.index_generators.Group.steps
```

````

`````

`````{py:class} IndicesForOperation(params: elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters, step_id, group_id)
:canonical: elasticai.creator_plugins.grouped_filter.src.index_generators.IndicesForOperation

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.index_generators.IndicesForOperation
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.index_generators.IndicesForOperation.__init__
```

````{py:method} __iter__()
:canonical: elasticai.creator_plugins.grouped_filter.src.index_generators.IndicesForOperation.__iter__

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.index_generators.IndicesForOperation.__iter__
```

````

`````

````{py:function} unrolled_grouped_convolution(spatial_input_size: int, window_size: int, in_channels: int, groups: int, stride: int)
:canonical: elasticai.creator_plugins.grouped_filter.src.index_generators.unrolled_grouped_convolution

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.index_generators.unrolled_grouped_convolution
```
````

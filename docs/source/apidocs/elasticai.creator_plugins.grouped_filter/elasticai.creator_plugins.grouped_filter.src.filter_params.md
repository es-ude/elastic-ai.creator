# {py:mod}`elasticai.creator_plugins.grouped_filter.src.filter_params`

```{py:module} elasticai.creator_plugins.grouped_filter.src.filter_params
```

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.filter_params
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FilterParameters <elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters>`
  - ```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters
    :summary:
    ```
````

### API

`````{py:class} FilterParameters(kernel_size: int, in_channels: int, out_channels: int, groups: int = 1, stride: int = 1, input_size: int | None = None, output_size: int = 1)
:canonical: elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters.__init__
```

````{py:attribute} _field_names
:canonical: elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters._field_names
:value: >
   ('out_channels', 'in_channels', 'kernel_size', 'groups', 'stride', 'input_size', 'output_size')

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters._field_names
```

````

````{py:method} _check_group_validity()
:canonical: elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters._check_group_validity

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters._check_group_validity
```

````

````{py:property} fan_in
:canonical: elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters.fan_in
:type: int

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters.fan_in
```

````

````{py:property} stride
:canonical: elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters.stride
:type: int

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters.stride
```

````

````{py:property} num_steps
:canonical: elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters.num_steps
:type: int

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters.num_steps
```

````

````{py:property} groups
:canonical: elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters.groups
:type: int

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters.groups
```

````

````{py:property} in_channels
:canonical: elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters.in_channels
:type: int

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters.in_channels
```

````

````{py:property} in_channels_per_group
:canonical: elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters.in_channels_per_group
:type: int

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters.in_channels_per_group
```

````

````{py:property} out_channels_per_group
:canonical: elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters.out_channels_per_group
:type: int

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters.out_channels_per_group
```

````

````{py:property} out_channels
:canonical: elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters.out_channels
:type: int

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters.out_channels
```

````

````{py:property} total_skipped_inputs_per_step
:canonical: elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters.total_skipped_inputs_per_step
:type: int

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters.total_skipped_inputs_per_step
```

````

````{py:method} get_in_channels_by_group() -> tuple[tuple[int, ...], ...]
:canonical: elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters.get_in_channels_by_group

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters.get_in_channels_by_group
```

````

````{py:method} get_out_channels_by_group() -> tuple[tuple[int, ...], ...]
:canonical: elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters.get_out_channels_by_group

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters.get_out_channels_by_group
```

````

````{py:method} _get_channels_by_group(channels_per_group)
:canonical: elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters._get_channels_by_group

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters._get_channels_by_group
```

````

````{py:method} with_groups(groups) -> elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters
:canonical: elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters.with_groups

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters.with_groups
```

````

````{py:method} _value_dict()
:canonical: elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters._value_dict

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters._value_dict
```

````

````{py:method} as_dict()
:canonical: elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters.as_dict

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters.as_dict
```

````

````{py:method} from_dict(d: dict)
:canonical: elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters.from_dict
:classmethod:

```{autodoc2-docstring} elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters.from_dict
```

````

````{py:method} __repr__() -> str
:canonical: elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters.__repr__

````

````{py:method} __hash__()
:canonical: elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters.__hash__

````

````{py:method} __eq__(other)
:canonical: elasticai.creator_plugins.grouped_filter.src.filter_params.FilterParameters.__eq__

````

`````

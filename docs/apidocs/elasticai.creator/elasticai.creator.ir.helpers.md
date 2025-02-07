# {py:mod}`elasticai.creator.ir.helpers`

```{py:module} elasticai.creator.ir.helpers
```

```{autodoc2-docstring} elasticai.creator.ir.helpers
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Shape <elasticai.creator.ir.helpers.Shape>`
  - ```{autodoc2-docstring} elasticai.creator.ir.helpers.Shape
    :summary:
    ```
* - {py:obj}`FilterParameters <elasticai.creator.ir.helpers.FilterParameters>`
  - ```{autodoc2-docstring} elasticai.creator.ir.helpers.FilterParameters
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`is_shape_tuple <elasticai.creator.ir.helpers.is_shape_tuple>`
  - ```{autodoc2-docstring} elasticai.creator.ir.helpers.is_shape_tuple
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ShapeTuple <elasticai.creator.ir.helpers.ShapeTuple>`
  - ```{autodoc2-docstring} elasticai.creator.ir.helpers.ShapeTuple
    :summary:
    ```
````

### API

````{py:data} ShapeTuple
:canonical: elasticai.creator.ir.helpers.ShapeTuple
:type: typing.TypeAlias
:value: >
   None

```{autodoc2-docstring} elasticai.creator.ir.helpers.ShapeTuple
```

````

````{py:function} is_shape_tuple(values) -> typing.TypeGuard[elasticai.creator.ir.helpers.ShapeTuple]
:canonical: elasticai.creator.ir.helpers.is_shape_tuple

```{autodoc2-docstring} elasticai.creator.ir.helpers.is_shape_tuple
```
````

`````{py:class} Shape(*values: int)
:canonical: elasticai.creator.ir.helpers.Shape

```{autodoc2-docstring} elasticai.creator.ir.helpers.Shape
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.ir.helpers.Shape.__init__
```

````{py:method} from_tuple(values: elasticai.creator.ir.helpers.ShapeTuple) -> elasticai.creator.ir.helpers.Shape
:canonical: elasticai.creator.ir.helpers.Shape.from_tuple
:classmethod:

```{autodoc2-docstring} elasticai.creator.ir.helpers.Shape.from_tuple
```

````

````{py:method} to_tuple() -> elasticai.creator.ir.helpers.ShapeTuple
:canonical: elasticai.creator.ir.helpers.Shape.to_tuple

```{autodoc2-docstring} elasticai.creator.ir.helpers.Shape.to_tuple
```

````

````{py:method} __getitem__(item)
:canonical: elasticai.creator.ir.helpers.Shape.__getitem__

```{autodoc2-docstring} elasticai.creator.ir.helpers.Shape.__getitem__
```

````

````{py:method} size() -> int
:canonical: elasticai.creator.ir.helpers.Shape.size

```{autodoc2-docstring} elasticai.creator.ir.helpers.Shape.size
```

````

````{py:method} ndim() -> int
:canonical: elasticai.creator.ir.helpers.Shape.ndim

```{autodoc2-docstring} elasticai.creator.ir.helpers.Shape.ndim
```

````

````{py:property} depth
:canonical: elasticai.creator.ir.helpers.Shape.depth
:type: int

```{autodoc2-docstring} elasticai.creator.ir.helpers.Shape.depth
```

````

````{py:method} __eq__(other)
:canonical: elasticai.creator.ir.helpers.Shape.__eq__

````

````{py:property} width
:canonical: elasticai.creator.ir.helpers.Shape.width
:type: int

```{autodoc2-docstring} elasticai.creator.ir.helpers.Shape.width
```

````

````{py:property} height
:canonical: elasticai.creator.ir.helpers.Shape.height
:type: int

```{autodoc2-docstring} elasticai.creator.ir.helpers.Shape.height
```

````

````{py:method} __repr__() -> str
:canonical: elasticai.creator.ir.helpers.Shape.__repr__

````

`````

`````{py:class} FilterParameters(kernel_size: int, in_channels: int, out_channels: int, groups: int = 1, stride: int = 1, input_size: int | None = None, output_size: int = 1)
:canonical: elasticai.creator.ir.helpers.FilterParameters

```{autodoc2-docstring} elasticai.creator.ir.helpers.FilterParameters
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.ir.helpers.FilterParameters.__init__
```

````{py:property} fan_in
:canonical: elasticai.creator.ir.helpers.FilterParameters.fan_in
:type: int

```{autodoc2-docstring} elasticai.creator.ir.helpers.FilterParameters.fan_in
```

````

````{py:property} stride
:canonical: elasticai.creator.ir.helpers.FilterParameters.stride
:type: int

```{autodoc2-docstring} elasticai.creator.ir.helpers.FilterParameters.stride
```

````

````{py:property} num_steps
:canonical: elasticai.creator.ir.helpers.FilterParameters.num_steps
:type: int

```{autodoc2-docstring} elasticai.creator.ir.helpers.FilterParameters.num_steps
```

````

````{py:property} groups
:canonical: elasticai.creator.ir.helpers.FilterParameters.groups
:type: int

```{autodoc2-docstring} elasticai.creator.ir.helpers.FilterParameters.groups
```

````

````{py:property} in_channels
:canonical: elasticai.creator.ir.helpers.FilterParameters.in_channels
:type: int

```{autodoc2-docstring} elasticai.creator.ir.helpers.FilterParameters.in_channels
```

````

````{py:property} in_channels_per_group
:canonical: elasticai.creator.ir.helpers.FilterParameters.in_channels_per_group
:type: int

```{autodoc2-docstring} elasticai.creator.ir.helpers.FilterParameters.in_channels_per_group
```

````

````{py:property} out_channels_per_group
:canonical: elasticai.creator.ir.helpers.FilterParameters.out_channels_per_group
:type: int

```{autodoc2-docstring} elasticai.creator.ir.helpers.FilterParameters.out_channels_per_group
```

````

````{py:property} out_channels
:canonical: elasticai.creator.ir.helpers.FilterParameters.out_channels
:type: int

```{autodoc2-docstring} elasticai.creator.ir.helpers.FilterParameters.out_channels
```

````

````{py:property} total_skipped_inputs_per_step
:canonical: elasticai.creator.ir.helpers.FilterParameters.total_skipped_inputs_per_step
:type: int

```{autodoc2-docstring} elasticai.creator.ir.helpers.FilterParameters.total_skipped_inputs_per_step
```

````

````{py:method} get_in_channels_by_group() -> tuple[tuple[int, ...], ...]
:canonical: elasticai.creator.ir.helpers.FilterParameters.get_in_channels_by_group

```{autodoc2-docstring} elasticai.creator.ir.helpers.FilterParameters.get_in_channels_by_group
```

````

````{py:method} get_out_channels_by_group() -> tuple[tuple[int, ...], ...]
:canonical: elasticai.creator.ir.helpers.FilterParameters.get_out_channels_by_group

```{autodoc2-docstring} elasticai.creator.ir.helpers.FilterParameters.get_out_channels_by_group
```

````

````{py:method} with_groups(groups) -> elasticai.creator.ir.helpers.FilterParameters
:canonical: elasticai.creator.ir.helpers.FilterParameters.with_groups

```{autodoc2-docstring} elasticai.creator.ir.helpers.FilterParameters.with_groups
```

````

````{py:method} as_dict()
:canonical: elasticai.creator.ir.helpers.FilterParameters.as_dict

```{autodoc2-docstring} elasticai.creator.ir.helpers.FilterParameters.as_dict
```

````

````{py:method} from_dict(d: dict)
:canonical: elasticai.creator.ir.helpers.FilterParameters.from_dict
:classmethod:

```{autodoc2-docstring} elasticai.creator.ir.helpers.FilterParameters.from_dict
```

````

````{py:method} __repr__() -> str
:canonical: elasticai.creator.ir.helpers.FilterParameters.__repr__

````

````{py:method} __hash__()
:canonical: elasticai.creator.ir.helpers.FilterParameters.__hash__

````

````{py:method} __eq__(other)
:canonical: elasticai.creator.ir.helpers.FilterParameters.__eq__

````

`````

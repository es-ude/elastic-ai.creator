# {py:mod}`elasticai.creator.nn.fixed_point.two_complement_fixed_point_config`

```{py:module} elasticai.creator.nn.fixed_point.two_complement_fixed_point_config
```

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.two_complement_fixed_point_config
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ConvertableToFixedPointValues <elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.ConvertableToFixedPointValues>`
  - ```{autodoc2-docstring} elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.ConvertableToFixedPointValues
    :summary:
    ```
* - {py:obj}`FixedPointConfig <elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.FixedPointConfig>`
  - ```{autodoc2-docstring} elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.FixedPointConfig
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`T <elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.T>`
  - ```{autodoc2-docstring} elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.T
    :summary:
    ```
````

### API

````{py:data} T
:canonical: elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.T
:value: >
   'TypeVar(...)'

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.T
```

````

`````{py:class} ConvertableToFixedPointValues
:canonical: elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.ConvertableToFixedPointValues

Bases: {py:obj}`typing.Protocol`\[{py:obj}`elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.T`\]

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.ConvertableToFixedPointValues
```

````{py:method} round() -> elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.T
:canonical: elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.ConvertableToFixedPointValues.round

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.ConvertableToFixedPointValues.round
```

````

````{py:method} int() -> elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.T
:canonical: elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.ConvertableToFixedPointValues.int

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.ConvertableToFixedPointValues.int
```

````

````{py:method} float() -> elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.T
:canonical: elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.ConvertableToFixedPointValues.float

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.ConvertableToFixedPointValues.float
```

````

````{py:method} __gt__(other: typing.Union[int, float, elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.T]) -> elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.T
:canonical: elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.ConvertableToFixedPointValues.__gt__

````

````{py:method} __lt__(other: typing.Union[int, float, elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.T]) -> elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.T
:canonical: elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.ConvertableToFixedPointValues.__lt__

````

````{py:method} __or__(other: elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.T) -> elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.T
:canonical: elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.ConvertableToFixedPointValues.__or__

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.ConvertableToFixedPointValues.__or__
```

````

````{py:method} __mul__(other: typing.Union[int, elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.T, float]) -> elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.T
:canonical: elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.ConvertableToFixedPointValues.__mul__

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.ConvertableToFixedPointValues.__mul__
```

````

````{py:method} __truediv__(other: typing.Union[int, float]) -> elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.T
:canonical: elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.ConvertableToFixedPointValues.__truediv__

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.ConvertableToFixedPointValues.__truediv__
```

````

`````

`````{py:class} FixedPointConfig
:canonical: elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.FixedPointConfig

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.FixedPointConfig
```

````{py:attribute} total_bits
:canonical: elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.FixedPointConfig.total_bits
:type: int
:value: >
   None

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.FixedPointConfig.total_bits
```

````

````{py:attribute} frac_bits
:canonical: elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.FixedPointConfig.frac_bits
:type: int
:value: >
   None

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.FixedPointConfig.frac_bits
```

````

````{py:property} minimum_as_integer
:canonical: elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.FixedPointConfig.minimum_as_integer
:type: int

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.FixedPointConfig.minimum_as_integer
```

````

````{py:property} maximum_as_integer
:canonical: elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.FixedPointConfig.maximum_as_integer
:type: int

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.FixedPointConfig.maximum_as_integer
```

````

````{py:property} minimum_as_rational
:canonical: elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.FixedPointConfig.minimum_as_rational
:type: float

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.FixedPointConfig.minimum_as_rational
```

````

````{py:property} maximum_as_rational
:canonical: elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.FixedPointConfig.maximum_as_rational
:type: float

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.FixedPointConfig.maximum_as_rational
```

````

````{py:method} integer_out_of_bounds(number: elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.T) -> elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.T
:canonical: elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.FixedPointConfig.integer_out_of_bounds

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.FixedPointConfig.integer_out_of_bounds
```

````

````{py:method} rational_out_of_bounds(number: elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.T) -> elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.T
:canonical: elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.FixedPointConfig.rational_out_of_bounds

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.FixedPointConfig.rational_out_of_bounds
```

````

````{py:method} as_integer(number: float | int | elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.T) -> int | elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.T
:canonical: elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.FixedPointConfig.as_integer

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.FixedPointConfig.as_integer
```

````

````{py:method} as_rational(number: float | int | elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.T) -> float | elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.T
:canonical: elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.FixedPointConfig.as_rational

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.FixedPointConfig.as_rational
```

````

`````

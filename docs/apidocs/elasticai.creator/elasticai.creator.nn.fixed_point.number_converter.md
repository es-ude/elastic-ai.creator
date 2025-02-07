# {py:mod}`elasticai.creator.nn.fixed_point.number_converter`

```{py:module} elasticai.creator.nn.fixed_point.number_converter
```

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.number_converter
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FXPParams <elasticai.creator.nn.fixed_point.number_converter.FXPParams>`
  - ```{autodoc2-docstring} elasticai.creator.nn.fixed_point.number_converter.FXPParams
    :summary:
    ```
* - {py:obj}`NumberConverter <elasticai.creator.nn.fixed_point.number_converter.NumberConverter>`
  - ```{autodoc2-docstring} elasticai.creator.nn.fixed_point.number_converter.NumberConverter
    :summary:
    ```
````

### API

`````{py:class} FXPParams
:canonical: elasticai.creator.nn.fixed_point.number_converter.FXPParams

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.number_converter.FXPParams
```

````{py:attribute} total_bits
:canonical: elasticai.creator.nn.fixed_point.number_converter.FXPParams.total_bits
:type: int
:value: >
   None

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.number_converter.FXPParams.total_bits
```

````

````{py:attribute} frac_bits
:canonical: elasticai.creator.nn.fixed_point.number_converter.FXPParams.frac_bits
:type: int
:value: >
   None

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.number_converter.FXPParams.frac_bits
```

````

`````

`````{py:class} NumberConverter(fxp_params: elasticai.creator.nn.fixed_point.number_converter.FXPParams)
:canonical: elasticai.creator.nn.fixed_point.number_converter.NumberConverter

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.number_converter.NumberConverter
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.number_converter.NumberConverter.__init__
```

````{py:method} bits_to_integer(pattern: str) -> int
:canonical: elasticai.creator.nn.fixed_point.number_converter.NumberConverter.bits_to_integer

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.number_converter.NumberConverter.bits_to_integer
```

````

````{py:method} bits_to_rational(pattern: str) -> float
:canonical: elasticai.creator.nn.fixed_point.number_converter.NumberConverter.bits_to_rational

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.number_converter.NumberConverter.bits_to_rational
```

````

````{py:method} rational_to_bits(rational: float) -> str
:canonical: elasticai.creator.nn.fixed_point.number_converter.NumberConverter.rational_to_bits

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.number_converter.NumberConverter.rational_to_bits
```

````

````{py:method} bits_to_natural(pattern: str) -> int
:canonical: elasticai.creator.nn.fixed_point.number_converter.NumberConverter.bits_to_natural

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.number_converter.NumberConverter.bits_to_natural
```

````

````{py:method} integer_to_bits(number: int) -> str
:canonical: elasticai.creator.nn.fixed_point.number_converter.NumberConverter.integer_to_bits

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.number_converter.NumberConverter.integer_to_bits
```

````

````{py:property} max_rational
:canonical: elasticai.creator.nn.fixed_point.number_converter.NumberConverter.max_rational
:type: float

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.number_converter.NumberConverter.max_rational
```

````

````{py:property} max_integer
:canonical: elasticai.creator.nn.fixed_point.number_converter.NumberConverter.max_integer
:type: int

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.number_converter.NumberConverter.max_integer
```

````

````{py:property} min_rational
:canonical: elasticai.creator.nn.fixed_point.number_converter.NumberConverter.min_rational
:type: float

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.number_converter.NumberConverter.min_rational
```

````

````{py:property} min_integer
:canonical: elasticai.creator.nn.fixed_point.number_converter.NumberConverter.min_integer
:type: int

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.number_converter.NumberConverter.min_integer
```

````

````{py:property} max_natural
:canonical: elasticai.creator.nn.fixed_point.number_converter.NumberConverter.max_natural
:type: int

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.number_converter.NumberConverter.max_natural
```

````

````{py:property} min_natural
:canonical: elasticai.creator.nn.fixed_point.number_converter.NumberConverter.min_natural
:type: int

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.number_converter.NumberConverter.min_natural
```

````

`````

# {py:mod}`elasticai.creator.nn.fixed_point.number_conversion`

```{py:module} elasticai.creator.nn.fixed_point.number_conversion
```

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.number_conversion
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`bits_to_integer <elasticai.creator.nn.fixed_point.number_conversion.bits_to_integer>`
  - ```{autodoc2-docstring} elasticai.creator.nn.fixed_point.number_conversion.bits_to_integer
    :summary:
    ```
* - {py:obj}`bits_to_rational <elasticai.creator.nn.fixed_point.number_conversion.bits_to_rational>`
  - ```{autodoc2-docstring} elasticai.creator.nn.fixed_point.number_conversion.bits_to_rational
    :summary:
    ```
* - {py:obj}`convert_rational_to_bit_pattern <elasticai.creator.nn.fixed_point.number_conversion.convert_rational_to_bit_pattern>`
  - ```{autodoc2-docstring} elasticai.creator.nn.fixed_point.number_conversion.convert_rational_to_bit_pattern
    :summary:
    ```
* - {py:obj}`bits_to_natural <elasticai.creator.nn.fixed_point.number_conversion.bits_to_natural>`
  - ```{autodoc2-docstring} elasticai.creator.nn.fixed_point.number_conversion.bits_to_natural
    :summary:
    ```
* - {py:obj}`integer_to_bits <elasticai.creator.nn.fixed_point.number_conversion.integer_to_bits>`
  - ```{autodoc2-docstring} elasticai.creator.nn.fixed_point.number_conversion.integer_to_bits
    :summary:
    ```
* - {py:obj}`max_rational <elasticai.creator.nn.fixed_point.number_conversion.max_rational>`
  - ```{autodoc2-docstring} elasticai.creator.nn.fixed_point.number_conversion.max_rational
    :summary:
    ```
* - {py:obj}`min_rational <elasticai.creator.nn.fixed_point.number_conversion.min_rational>`
  - ```{autodoc2-docstring} elasticai.creator.nn.fixed_point.number_conversion.min_rational
    :summary:
    ```
* - {py:obj}`min_integer <elasticai.creator.nn.fixed_point.number_conversion.min_integer>`
  - ```{autodoc2-docstring} elasticai.creator.nn.fixed_point.number_conversion.min_integer
    :summary:
    ```
* - {py:obj}`max_integer <elasticai.creator.nn.fixed_point.number_conversion.max_integer>`
  - ```{autodoc2-docstring} elasticai.creator.nn.fixed_point.number_conversion.max_integer
    :summary:
    ```
* - {py:obj}`min_natural <elasticai.creator.nn.fixed_point.number_conversion.min_natural>`
  - ```{autodoc2-docstring} elasticai.creator.nn.fixed_point.number_conversion.min_natural
    :summary:
    ```
* - {py:obj}`max_natural <elasticai.creator.nn.fixed_point.number_conversion.max_natural>`
  - ```{autodoc2-docstring} elasticai.creator.nn.fixed_point.number_conversion.max_natural
    :summary:
    ```
````

### API

````{py:function} bits_to_integer(pattern: str) -> int
:canonical: elasticai.creator.nn.fixed_point.number_conversion.bits_to_integer

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.number_conversion.bits_to_integer
```
````

````{py:function} bits_to_rational(pattern: str, frac_bits: int) -> float
:canonical: elasticai.creator.nn.fixed_point.number_conversion.bits_to_rational

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.number_conversion.bits_to_rational
```
````

````{py:function} convert_rational_to_bit_pattern(rational: float, total_bits: int, frac_bits: int) -> str
:canonical: elasticai.creator.nn.fixed_point.number_conversion.convert_rational_to_bit_pattern

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.number_conversion.convert_rational_to_bit_pattern
```
````

````{py:function} bits_to_natural(pattern: str) -> int
:canonical: elasticai.creator.nn.fixed_point.number_conversion.bits_to_natural

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.number_conversion.bits_to_natural
```
````

````{py:function} integer_to_bits(number: int, total_bits: int) -> str
:canonical: elasticai.creator.nn.fixed_point.number_conversion.integer_to_bits

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.number_conversion.integer_to_bits
```
````

````{py:function} max_rational(total_bits: int, frac_bits: int) -> float
:canonical: elasticai.creator.nn.fixed_point.number_conversion.max_rational

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.number_conversion.max_rational
```
````

````{py:function} min_rational(total_bits: int, frac_bits: int) -> float
:canonical: elasticai.creator.nn.fixed_point.number_conversion.min_rational

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.number_conversion.min_rational
```
````

````{py:function} min_integer(total_bits: int) -> int
:canonical: elasticai.creator.nn.fixed_point.number_conversion.min_integer

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.number_conversion.min_integer
```
````

````{py:function} max_integer(total_bits: int) -> int
:canonical: elasticai.creator.nn.fixed_point.number_conversion.max_integer

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.number_conversion.max_integer
```
````

````{py:function} min_natural(total_bits: int) -> int
:canonical: elasticai.creator.nn.fixed_point.number_conversion.min_natural

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.number_conversion.min_natural
```
````

````{py:function} max_natural(total_bits: int) -> int
:canonical: elasticai.creator.nn.fixed_point.number_conversion.max_natural

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.number_conversion.max_natural
```
````

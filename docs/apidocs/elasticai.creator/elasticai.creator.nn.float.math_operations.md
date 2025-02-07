# {py:mod}`elasticai.creator.nn.float.math_operations`

```{py:module} elasticai.creator.nn.float.math_operations
```

```{autodoc2-docstring} elasticai.creator.nn.float.math_operations
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MathOperations <elasticai.creator.nn.float.math_operations.MathOperations>`
  -
````

### API

`````{py:class} MathOperations(mantissa_bits: int, exponent_bits: int)
:canonical: elasticai.creator.nn.float.math_operations.MathOperations

Bases: {py:obj}`elasticai.creator.base_modules.linear.MathOperations`, {py:obj}`elasticai.creator.base_modules.conv1d.MathOperations`, {py:obj}`elasticai.creator.base_modules.lstm_cell.MathOperations`

````{py:property} largest_positive_value
:canonical: elasticai.creator.nn.float.math_operations.MathOperations.largest_positive_value
:type: float

```{autodoc2-docstring} elasticai.creator.nn.float.math_operations.MathOperations.largest_positive_value
```

````

````{py:property} smallest_negative_value
:canonical: elasticai.creator.nn.float.math_operations.MathOperations.smallest_negative_value
:type: float

```{autodoc2-docstring} elasticai.creator.nn.float.math_operations.MathOperations.smallest_negative_value
```

````

````{py:method} quantize(a: torch.Tensor) -> torch.Tensor
:canonical: elasticai.creator.nn.float.math_operations.MathOperations.quantize

```{autodoc2-docstring} elasticai.creator.nn.float.math_operations.MathOperations.quantize
```

````

````{py:method} add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor
:canonical: elasticai.creator.nn.float.math_operations.MathOperations.add

```{autodoc2-docstring} elasticai.creator.nn.float.math_operations.MathOperations.add
```

````

````{py:method} matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor
:canonical: elasticai.creator.nn.float.math_operations.MathOperations.matmul

```{autodoc2-docstring} elasticai.creator.nn.float.math_operations.MathOperations.matmul
```

````

````{py:method} mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor
:canonical: elasticai.creator.nn.float.math_operations.MathOperations.mul

```{autodoc2-docstring} elasticai.creator.nn.float.math_operations.MathOperations.mul
```

````

`````

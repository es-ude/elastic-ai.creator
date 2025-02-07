# {py:mod}`elasticai.creator.nn.fixed_point.math_operations`

```{py:module} elasticai.creator.nn.fixed_point.math_operations
```

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.math_operations
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MathOperations <elasticai.creator.nn.fixed_point.math_operations.MathOperations>`
  -
````

### API

`````{py:class} MathOperations(config: elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.FixedPointConfig)
:canonical: elasticai.creator.nn.fixed_point.math_operations.MathOperations

Bases: {py:obj}`elasticai.creator.base_modules.linear.MathOperations`, {py:obj}`elasticai.creator.base_modules.conv1d.MathOperations`, {py:obj}`elasticai.creator.base_modules.lstm_cell.MathOperations`

````{py:method} quantize(a: torch.Tensor) -> torch.Tensor
:canonical: elasticai.creator.nn.fixed_point.math_operations.MathOperations.quantize

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.math_operations.MathOperations.quantize
```

````

````{py:method} add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor
:canonical: elasticai.creator.nn.fixed_point.math_operations.MathOperations.add

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.math_operations.MathOperations.add
```

````

````{py:method} matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor
:canonical: elasticai.creator.nn.fixed_point.math_operations.MathOperations.matmul

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.math_operations.MathOperations.matmul
```

````

````{py:method} mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor
:canonical: elasticai.creator.nn.fixed_point.math_operations.MathOperations.mul

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.math_operations.MathOperations.mul
```

````

`````

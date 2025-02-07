# {py:mod}`elasticai.creator.base_modules.linear`

```{py:module} elasticai.creator.base_modules.linear
```

```{autodoc2-docstring} elasticai.creator.base_modules.linear
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MathOperations <elasticai.creator.base_modules.linear.MathOperations>`
  -
* - {py:obj}`Linear <elasticai.creator.base_modules.linear.Linear>`
  -
````

### API

```{py:class} MathOperations
:canonical: elasticai.creator.base_modules.linear.MathOperations

Bases: {py:obj}`elasticai.creator.base_modules.math_operations.Quantize`, {py:obj}`elasticai.creator.base_modules.math_operations.Add`, {py:obj}`elasticai.creator.base_modules.math_operations.MatMul`, {py:obj}`typing.Protocol`

```

`````{py:class} Linear(in_features: int, out_features: int, operations: elasticai.creator.base_modules.linear.MathOperations, bias: bool, device: typing.Any = None, dtype: typing.Any = None)
:canonical: elasticai.creator.base_modules.linear.Linear

Bases: {py:obj}`torch.nn.Linear`

````{py:method} forward(x: torch.Tensor) -> torch.Tensor
:canonical: elasticai.creator.base_modules.linear.Linear.forward

```{autodoc2-docstring} elasticai.creator.base_modules.linear.Linear.forward
```

````

`````

# {py:mod}`elasticai.creator.base_modules.math_operations`

```{py:module} elasticai.creator.base_modules.math_operations
```

```{autodoc2-docstring} elasticai.creator.base_modules.math_operations
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Quantize <elasticai.creator.base_modules.math_operations.Quantize>`
  -
* - {py:obj}`MatMul <elasticai.creator.base_modules.math_operations.MatMul>`
  -
* - {py:obj}`Add <elasticai.creator.base_modules.math_operations.Add>`
  -
* - {py:obj}`Mul <elasticai.creator.base_modules.math_operations.Mul>`
  -
````

### API

`````{py:class} Quantize
:canonical: elasticai.creator.base_modules.math_operations.Quantize

Bases: {py:obj}`typing.Protocol`

````{py:method} quantize(a: torch.Tensor) -> torch.Tensor
:canonical: elasticai.creator.base_modules.math_operations.Quantize.quantize
:abstractmethod:

```{autodoc2-docstring} elasticai.creator.base_modules.math_operations.Quantize.quantize
```

````

`````

`````{py:class} MatMul
:canonical: elasticai.creator.base_modules.math_operations.MatMul

Bases: {py:obj}`typing.Protocol`

````{py:method} matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor
:canonical: elasticai.creator.base_modules.math_operations.MatMul.matmul
:abstractmethod:

```{autodoc2-docstring} elasticai.creator.base_modules.math_operations.MatMul.matmul
```

````

`````

`````{py:class} Add
:canonical: elasticai.creator.base_modules.math_operations.Add

Bases: {py:obj}`typing.Protocol`

````{py:method} add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor
:canonical: elasticai.creator.base_modules.math_operations.Add.add
:abstractmethod:

```{autodoc2-docstring} elasticai.creator.base_modules.math_operations.Add.add
```

````

`````

`````{py:class} Mul
:canonical: elasticai.creator.base_modules.math_operations.Mul

Bases: {py:obj}`typing.Protocol`

````{py:method} mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor
:canonical: elasticai.creator.base_modules.math_operations.Mul.mul
:abstractmethod:

```{autodoc2-docstring} elasticai.creator.base_modules.math_operations.Mul.mul
```

````

`````

# {py:mod}`elasticai.creator.base_modules.conv1d`

```{py:module} elasticai.creator.base_modules.conv1d
```

```{autodoc2-docstring} elasticai.creator.base_modules.conv1d
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MathOperations <elasticai.creator.base_modules.conv1d.MathOperations>`
  -
* - {py:obj}`Conv1d <elasticai.creator.base_modules.conv1d.Conv1d>`
  -
````

### API

```{py:class} MathOperations
:canonical: elasticai.creator.base_modules.conv1d.MathOperations

Bases: {py:obj}`elasticai.creator.base_modules.math_operations.Quantize`, {py:obj}`typing.Protocol`

```

`````{py:class} Conv1d(operations: elasticai.creator.base_modules.conv1d.MathOperations, in_channels: int, out_channels: int, kernel_size: int | tuple[int], stride: int | tuple[int] = 1, padding: int | tuple[int] | str = 0, dilation: int | tuple[int] = 1, groups: int = 1, bias: bool = True, device: typing.Any = None, dtype: typing.Any = None)
:canonical: elasticai.creator.base_modules.conv1d.Conv1d

Bases: {py:obj}`torch.nn.Conv1d`

````{py:method} forward(x: torch.Tensor) -> torch.Tensor
:canonical: elasticai.creator.base_modules.conv1d.Conv1d.forward

```{autodoc2-docstring} elasticai.creator.base_modules.conv1d.Conv1d.forward
```

````

`````

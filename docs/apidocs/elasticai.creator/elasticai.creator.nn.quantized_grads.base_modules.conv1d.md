# {py:mod}`elasticai.creator.nn.quantized_grads.base_modules.conv1d`

```{py:module} elasticai.creator.nn.quantized_grads.base_modules.conv1d
```

```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.base_modules.conv1d
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Conv1d <elasticai.creator.nn.quantized_grads.base_modules.conv1d.Conv1d>`
  - ```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.base_modules.conv1d.Conv1d
    :summary:
    ```
````

### API

`````{py:class} Conv1d(math_ops: torch.nn.Module, weight_quantization: torch.nn.Module, in_channels: int, out_channels: int, kernel_size: int | tuple[int], stride: int | tuple[int] = 1, padding: int | tuple[int] | str = 0, dilation: int | tuple[int] = 1, groups: int = 1, bias: bool = True, bias_quantization: torch.nn.Module = None, device: typing.Any = None, dtype: typing.Any = None)
:canonical: elasticai.creator.nn.quantized_grads.base_modules.conv1d.Conv1d

Bases: {py:obj}`torch.nn.Conv1d`

```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.base_modules.conv1d.Conv1d
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.base_modules.conv1d.Conv1d.__init__
```

````{py:method} forward(x: torch.Tensor) -> torch.Tensor
:canonical: elasticai.creator.nn.quantized_grads.base_modules.conv1d.Conv1d.forward

```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.base_modules.conv1d.Conv1d.forward
```

````

`````

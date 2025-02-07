# {py:mod}`elasticai.creator.nn.quantized_grads.base_modules.conv2d`

```{py:module} elasticai.creator.nn.quantized_grads.base_modules.conv2d
```

```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.base_modules.conv2d
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Conv2d <elasticai.creator.nn.quantized_grads.base_modules.conv2d.Conv2d>`
  - ```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.base_modules.conv2d.Conv2d
    :summary:
    ```
````

### API

`````{py:class} Conv2d(math_ops: torch.nn.Module, weight_quantization: torch.nn.Module, in_channels: int, out_channels: int, kernel_size: int | tuple[int], stride: int | tuple[int] = 1, padding: int | tuple[int] | str = 0, dilation: int | tuple[int] = 1, groups: int = 1, bias: bool = True, bias_quantization: torch.nn.Module = None, device: typing.Any = None, dtype: typing.Any = None)
:canonical: elasticai.creator.nn.quantized_grads.base_modules.conv2d.Conv2d

Bases: {py:obj}`torch.nn.Conv2d`

```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.base_modules.conv2d.Conv2d
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.base_modules.conv2d.Conv2d.__init__
```

````{py:method} forward(x: torch.Tensor) -> torch.Tensor
:canonical: elasticai.creator.nn.quantized_grads.base_modules.conv2d.Conv2d.forward

```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.base_modules.conv2d.Conv2d.forward
```

````

`````

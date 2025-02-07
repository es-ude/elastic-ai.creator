# {py:mod}`elasticai.creator.nn.quantized_grads.base_modules.batchnorm2d`

```{py:module} elasticai.creator.nn.quantized_grads.base_modules.batchnorm2d
```

```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.base_modules.batchnorm2d
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BatchNorm2d <elasticai.creator.nn.quantized_grads.base_modules.batchnorm2d.BatchNorm2d>`
  - ```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.base_modules.batchnorm2d.BatchNorm2d
    :summary:
    ```
````

### API

`````{py:class} BatchNorm2d(math_ops: torch.nn.Module, weight_quantization: torch.nn.Module, bias_quantization: torch.nn.Module, num_features: int, eps: float = 1e-05, momentum: float = 0.1, affine: bool = True, track_running_stats: bool = True, device: typing.Any = None, dtype: typing.Any = None)
:canonical: elasticai.creator.nn.quantized_grads.base_modules.batchnorm2d.BatchNorm2d

Bases: {py:obj}`torch.nn.BatchNorm2d`

```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.base_modules.batchnorm2d.BatchNorm2d
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.base_modules.batchnorm2d.BatchNorm2d.__init__
```

````{py:method} forward(x: torch.Tensor) -> torch.Tensor
:canonical: elasticai.creator.nn.quantized_grads.base_modules.batchnorm2d.BatchNorm2d.forward

```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.base_modules.batchnorm2d.BatchNorm2d.forward
```

````

`````

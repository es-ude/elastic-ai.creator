# {py:mod}`elasticai.creator.nn.quantized_grads.base_modules.linear`

```{py:module} elasticai.creator.nn.quantized_grads.base_modules.linear
```

```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.base_modules.linear
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Linear <elasticai.creator.nn.quantized_grads.base_modules.linear.Linear>`
  - ```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.base_modules.linear.Linear
    :summary:
    ```
````

### API

`````{py:class} Linear(math_ops: torch.nn.Module, in_features: int, out_features: int, weight_quantization: torch.nn.Module, bias: bool, bias_quantization: torch.nn.Module = None, device: typing.Any = None, dtype: typing.Any = None)
:canonical: elasticai.creator.nn.quantized_grads.base_modules.linear.Linear

Bases: {py:obj}`torch.nn.Linear`

```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.base_modules.linear.Linear
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.base_modules.linear.Linear.__init__
```

````{py:method} forward(x: torch.Tensor) -> torch.Tensor
:canonical: elasticai.creator.nn.quantized_grads.base_modules.linear.Linear.forward

```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.base_modules.linear.Linear.forward
```

````

`````

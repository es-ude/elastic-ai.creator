# {py:mod}`elasticai.creator.nn.quantized_grads`

```{py:module} elasticai.creator.nn.quantized_grads
```

```{autodoc2-docstring} elasticai.creator.nn.quantized_grads
:allowtitles:
```

## Package Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Identity <elasticai.creator.nn.quantized_grads.identity_param_quantization.Identity>`
  -
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_quantized_optimizer <elasticai.creator.nn.quantized_grads.quantized_optim.get_quantized_optimizer>`
  - ```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.quantized_optim.get_quantized_optimizer
    :summary:
    ```
````

### API

````{py:function} get_quantized_optimizer(optimizer: type[torch.optim.Optimizer]) -> type[elasticai.creator.nn.quantized_grads.quantized_optim._QOptim]
:canonical: elasticai.creator.nn.quantized_grads.quantized_optim.get_quantized_optimizer

```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.quantized_optim.get_quantized_optimizer
```
````

`````{py:class} Identity(*args: typing.Any, **kwargs: typing.Any)
:canonical: elasticai.creator.nn.quantized_grads.identity_param_quantization.Identity

Bases: {py:obj}`torch.nn.Identity`

````{py:method} right_inverse(x: torch.Tensor) -> torch.Tensor
:canonical: elasticai.creator.nn.quantized_grads.identity_param_quantization.Identity.right_inverse
:staticmethod:

```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.identity_param_quantization.Identity.right_inverse
```

````

`````

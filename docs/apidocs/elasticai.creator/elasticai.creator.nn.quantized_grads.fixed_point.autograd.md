# {py:mod}`elasticai.creator.nn.quantized_grads.fixed_point.autograd`

```{py:module} elasticai.creator.nn.quantized_grads.fixed_point.autograd
```

```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.fixed_point.autograd
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_autograd_func <elasticai.creator.nn.quantized_grads.fixed_point.autograd.get_autograd_func>`
  - ```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.fixed_point.autograd.get_autograd_func
    :summary:
    ```
````

### API

````{py:function} get_autograd_func(forw_func: typing.Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor], backw_func: typing.Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]) -> tuple[type[torch.autograd.Function], type[torch.autograd.Function]]
:canonical: elasticai.creator.nn.quantized_grads.fixed_point.autograd.get_autograd_func

```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.fixed_point.autograd.get_autograd_func
```
````

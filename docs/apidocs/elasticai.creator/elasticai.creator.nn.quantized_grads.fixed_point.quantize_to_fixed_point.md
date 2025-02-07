# {py:mod}`elasticai.creator.nn.quantized_grads.fixed_point.quantize_to_fixed_point`

```{py:module} elasticai.creator.nn.quantized_grads.fixed_point.quantize_to_fixed_point
```

```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.fixed_point.quantize_to_fixed_point
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Round <elasticai.creator.nn.quantized_grads.fixed_point.quantize_to_fixed_point.Round>`
  - ```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.fixed_point.quantize_to_fixed_point.Round
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`round_tensor <elasticai.creator.nn.quantized_grads.fixed_point.quantize_to_fixed_point.round_tensor>`
  - ```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.fixed_point.quantize_to_fixed_point.round_tensor
    :summary:
    ```
* - {py:obj}`quantize_to_fxp_hte <elasticai.creator.nn.quantized_grads.fixed_point.quantize_to_fixed_point.quantize_to_fxp_hte>`
  - ```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.fixed_point.quantize_to_fixed_point.quantize_to_fxp_hte
    :summary:
    ```
* - {py:obj}`quantize_to_fxp_stochastic <elasticai.creator.nn.quantized_grads.fixed_point.quantize_to_fixed_point.quantize_to_fxp_stochastic>`
  - ```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.fixed_point.quantize_to_fixed_point.quantize_to_fxp_stochastic
    :summary:
    ```
````

### API

`````{py:class} Round(*args, **kwargs)
:canonical: elasticai.creator.nn.quantized_grads.fixed_point.quantize_to_fixed_point.Round

Bases: {py:obj}`torch.autograd.Function`

```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.fixed_point.quantize_to_fixed_point.Round
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.fixed_point.quantize_to_fixed_point.Round.__init__
```

````{py:method} forward(ctx, x, *args, **kwargs)
:canonical: elasticai.creator.nn.quantized_grads.fixed_point.quantize_to_fixed_point.Round.forward
:staticmethod:

````

````{py:method} backward(ctx, *grad_output)
:canonical: elasticai.creator.nn.quantized_grads.fixed_point.quantize_to_fixed_point.Round.backward
:staticmethod:

````

`````

````{py:function} round_tensor(x: torch.Tensor) -> torch.Tensor
:canonical: elasticai.creator.nn.quantized_grads.fixed_point.quantize_to_fixed_point.round_tensor

```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.fixed_point.quantize_to_fixed_point.round_tensor
```
````

````{py:function} quantize_to_fxp_hte(number: torch.Tensor, resolution_per_int: torch.Tensor, minimum_as_rational: torch.Tensor, maximum_as_rational: torch.Tensor) -> torch.Tensor
:canonical: elasticai.creator.nn.quantized_grads.fixed_point.quantize_to_fixed_point.quantize_to_fxp_hte

```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.fixed_point.quantize_to_fixed_point.quantize_to_fxp_hte
```
````

````{py:function} quantize_to_fxp_stochastic(number: torch.Tensor, resolution_per_int: torch.Tensor, minimum_as_rational: torch.Tensor, maximum_as_rational: torch.Tensor) -> torch.Tensor
:canonical: elasticai.creator.nn.quantized_grads.fixed_point.quantize_to_fixed_point.quantize_to_fxp_stochastic

```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.fixed_point.quantize_to_fixed_point.quantize_to_fxp_stochastic
```
````

# {py:mod}`elasticai.creator.nn.quantized_grads.fixed_point.param_quantization`

```{py:module} elasticai.creator.nn.quantized_grads.fixed_point.param_quantization
```

```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.fixed_point.param_quantization
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`QuantizeTensorToFixedPointHTE <elasticai.creator.nn.quantized_grads.fixed_point.param_quantization.QuantizeTensorToFixedPointHTE>`
  - ```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.fixed_point.param_quantization.QuantizeTensorToFixedPointHTE
    :summary:
    ```
* - {py:obj}`QuantizeTensorToFixedPointStochastic <elasticai.creator.nn.quantized_grads.fixed_point.param_quantization.QuantizeTensorToFixedPointStochastic>`
  - ```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.fixed_point.param_quantization.QuantizeTensorToFixedPointStochastic
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_quantize_to_fixed_point <elasticai.creator.nn.quantized_grads.fixed_point.param_quantization.get_quantize_to_fixed_point>`
  - ```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.fixed_point.param_quantization.get_quantize_to_fixed_point
    :summary:
    ```
````

### API

````{py:function} get_quantize_to_fixed_point(func: typing.Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]) -> tuple[type[torch.nn.Module], type[torch.nn.Module]]
:canonical: elasticai.creator.nn.quantized_grads.fixed_point.param_quantization.get_quantize_to_fixed_point

```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.fixed_point.param_quantization.get_quantize_to_fixed_point
```
````

````{py:class} QuantizeTensorToFixedPointHTE(config: elasticai.creator.nn.quantized_grads.fixed_point.two_complement_fixed_point_config.FixedPointConfigV2)
:canonical: elasticai.creator.nn.quantized_grads.fixed_point.param_quantization.QuantizeTensorToFixedPointHTE

Bases: {py:obj}`elasticai.creator.nn.quantized_grads.fixed_point.param_quantization.QuantizeParamSTEToFixedPointHTE`

```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.fixed_point.param_quantization.QuantizeTensorToFixedPointHTE
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.fixed_point.param_quantization.QuantizeTensorToFixedPointHTE.__init__
```

````

````{py:class} QuantizeTensorToFixedPointStochastic(config: elasticai.creator.nn.quantized_grads.fixed_point.two_complement_fixed_point_config.FixedPointConfigV2)
:canonical: elasticai.creator.nn.quantized_grads.fixed_point.param_quantization.QuantizeTensorToFixedPointStochastic

Bases: {py:obj}`elasticai.creator.nn.quantized_grads.fixed_point.param_quantization.QuantizeParamSTEToFixedPointStochastic`

```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.fixed_point.param_quantization.QuantizeTensorToFixedPointStochastic
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.nn.quantized_grads.fixed_point.param_quantization.QuantizeTensorToFixedPointStochastic.__init__
```

````

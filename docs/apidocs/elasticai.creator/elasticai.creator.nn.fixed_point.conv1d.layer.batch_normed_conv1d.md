# {py:mod}`elasticai.creator.nn.fixed_point.conv1d.layer.batch_normed_conv1d`

```{py:module} elasticai.creator.nn.fixed_point.conv1d.layer.batch_normed_conv1d
```

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.conv1d.layer.batch_normed_conv1d
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BatchNormedConv1d <elasticai.creator.nn.fixed_point.conv1d.layer.batch_normed_conv1d.BatchNormedConv1d>`
  -
````

### API

`````{py:class} BatchNormedConv1d(total_bits: int, frac_bits: int, in_channels: int, out_channels: int, signal_length: int, kernel_size: int | tuple[int], bn_eps: float = 1e-05, bn_momentum: float = 0.1, bn_affine: bool = True, stride: int | tuple[int] = 1, padding: int | tuple[int] = 0, bias: bool = True, device: typing.Any = None)
:canonical: elasticai.creator.nn.fixed_point.conv1d.layer.batch_normed_conv1d.BatchNormedConv1d

Bases: {py:obj}`elasticai.creator.nn.design_creator_module.DesignCreatorModule`

````{py:property} conv_weight
:canonical: elasticai.creator.nn.fixed_point.conv1d.layer.batch_normed_conv1d.BatchNormedConv1d.conv_weight
:type: torch.Tensor

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.conv1d.layer.batch_normed_conv1d.BatchNormedConv1d.conv_weight
```

````

````{py:property} conv_bias
:canonical: elasticai.creator.nn.fixed_point.conv1d.layer.batch_normed_conv1d.BatchNormedConv1d.conv_bias
:type: torch.Tensor | None

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.conv1d.layer.batch_normed_conv1d.BatchNormedConv1d.conv_bias
```

````

````{py:property} bn_weight
:canonical: elasticai.creator.nn.fixed_point.conv1d.layer.batch_normed_conv1d.BatchNormedConv1d.bn_weight
:type: torch.Tensor

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.conv1d.layer.batch_normed_conv1d.BatchNormedConv1d.bn_weight
```

````

````{py:property} bn_bias
:canonical: elasticai.creator.nn.fixed_point.conv1d.layer.batch_normed_conv1d.BatchNormedConv1d.bn_bias
:type: torch.Tensor

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.conv1d.layer.batch_normed_conv1d.BatchNormedConv1d.bn_bias
```

````

````{py:method} forward(x: torch.Tensor) -> torch.Tensor
:canonical: elasticai.creator.nn.fixed_point.conv1d.layer.batch_normed_conv1d.BatchNormedConv1d.forward

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.conv1d.layer.batch_normed_conv1d.BatchNormedConv1d.forward
```

````

````{py:method} create_design(name: str) -> elasticai.creator.nn.fixed_point.conv1d.design.Conv1dDesign
:canonical: elasticai.creator.nn.fixed_point.conv1d.layer.batch_normed_conv1d.BatchNormedConv1d.create_design

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.conv1d.layer.batch_normed_conv1d.BatchNormedConv1d.create_design
```

````

`````

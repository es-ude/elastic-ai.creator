# {py:mod}`elasticai.creator.nn.fixed_point.linear.layer.batch_normed_linear`

```{py:module} elasticai.creator.nn.fixed_point.linear.layer.batch_normed_linear
```

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.linear.layer.batch_normed_linear
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BatchNormedLinear <elasticai.creator.nn.fixed_point.linear.layer.batch_normed_linear.BatchNormedLinear>`
  -
````

### API

`````{py:class} BatchNormedLinear(total_bits: int, frac_bits: int, in_features: int, out_features: int, bias: bool = True, bn_eps: float = 1e-05, bn_momentum: float = 0.1, bn_affine: bool = True, device: typing.Any = None)
:canonical: elasticai.creator.nn.fixed_point.linear.layer.batch_normed_linear.BatchNormedLinear

Bases: {py:obj}`elasticai.creator.nn.design_creator_module.DesignCreatorModule`, {py:obj}`torch.nn.Module`

````{py:property} lin_weight
:canonical: elasticai.creator.nn.fixed_point.linear.layer.batch_normed_linear.BatchNormedLinear.lin_weight
:type: torch.Tensor

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.linear.layer.batch_normed_linear.BatchNormedLinear.lin_weight
```

````

````{py:property} lin_bias
:canonical: elasticai.creator.nn.fixed_point.linear.layer.batch_normed_linear.BatchNormedLinear.lin_bias
:type: torch.Tensor | None

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.linear.layer.batch_normed_linear.BatchNormedLinear.lin_bias
```

````

````{py:property} bn_weight
:canonical: elasticai.creator.nn.fixed_point.linear.layer.batch_normed_linear.BatchNormedLinear.bn_weight
:type: torch.Tensor

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.linear.layer.batch_normed_linear.BatchNormedLinear.bn_weight
```

````

````{py:property} bn_bias
:canonical: elasticai.creator.nn.fixed_point.linear.layer.batch_normed_linear.BatchNormedLinear.bn_bias
:type: torch.Tensor

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.linear.layer.batch_normed_linear.BatchNormedLinear.bn_bias
```

````

````{py:method} forward(x: torch.Tensor) -> torch.Tensor
:canonical: elasticai.creator.nn.fixed_point.linear.layer.batch_normed_linear.BatchNormedLinear.forward

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.linear.layer.batch_normed_linear.BatchNormedLinear.forward
```

````

````{py:method} create_design(name: str) -> elasticai.creator.nn.fixed_point.linear.design.LinearDesign
:canonical: elasticai.creator.nn.fixed_point.linear.layer.batch_normed_linear.BatchNormedLinear.create_design

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.linear.layer.batch_normed_linear.BatchNormedLinear.create_design
```

````

`````

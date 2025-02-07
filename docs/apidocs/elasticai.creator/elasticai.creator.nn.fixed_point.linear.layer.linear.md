# {py:mod}`elasticai.creator.nn.fixed_point.linear.layer.linear`

```{py:module} elasticai.creator.nn.fixed_point.linear.layer.linear
```

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.linear.layer.linear
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Linear <elasticai.creator.nn.fixed_point.linear.layer.linear.Linear>`
  -
````

### API

`````{py:class} Linear(in_features: int, out_features: int, total_bits: int, frac_bits: int, bias: bool = True, device: typing.Any = None)
:canonical: elasticai.creator.nn.fixed_point.linear.layer.linear.Linear

Bases: {py:obj}`elasticai.creator.nn.design_creator_module.DesignCreatorModule`, {py:obj}`elasticai.creator.base_modules.linear.Linear`

````{py:method} create_design(name: str) -> elasticai.creator.nn.fixed_point.linear.design.LinearDesign
:canonical: elasticai.creator.nn.fixed_point.linear.layer.linear.Linear.create_design

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.linear.layer.linear.Linear.create_design
```

````

````{py:method} create_testbench(name: str, uut: elasticai.creator.nn.fixed_point.linear.design.LinearDesign) -> elasticai.creator.nn.fixed_point.linear.testbench.LinearTestbench
:canonical: elasticai.creator.nn.fixed_point.linear.layer.linear.Linear.create_testbench

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.linear.layer.linear.Linear.create_testbench
```

````

`````

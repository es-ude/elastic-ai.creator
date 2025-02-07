# {py:mod}`elasticai.creator.nn.fixed_point.hard_tanh.layer`

```{py:module} elasticai.creator.nn.fixed_point.hard_tanh.layer
```

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.hard_tanh.layer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HardTanh <elasticai.creator.nn.fixed_point.hard_tanh.layer.HardTanh>`
  -
````

### API

`````{py:class} HardTanh(total_bits: int, frac_bits: int, min_val: float = -1, max_val: float = 1)
:canonical: elasticai.creator.nn.fixed_point.hard_tanh.layer.HardTanh

Bases: {py:obj}`elasticai.creator.nn.design_creator_module.DesignCreatorModule`, {py:obj}`elasticai.creator.base_modules.hard_tanh.HardTanh`

````{py:method} create_design(name: str) -> elasticai.creator.nn.fixed_point.hard_tanh.design.HardTanh
:canonical: elasticai.creator.nn.fixed_point.hard_tanh.layer.HardTanh.create_design

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.hard_tanh.layer.HardTanh.create_design
```

````

`````

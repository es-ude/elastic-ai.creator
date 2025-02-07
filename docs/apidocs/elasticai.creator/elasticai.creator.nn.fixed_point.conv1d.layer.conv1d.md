# {py:mod}`elasticai.creator.nn.fixed_point.conv1d.layer.conv1d`

```{py:module} elasticai.creator.nn.fixed_point.conv1d.layer.conv1d
```

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.conv1d.layer.conv1d
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Conv1d <elasticai.creator.nn.fixed_point.conv1d.layer.conv1d.Conv1d>`
  -
````

### API

`````{py:class} Conv1d(total_bits: int, frac_bits: int, in_channels: int, out_channels: int, signal_length: int, kernel_size: int | tuple[int], bias: bool = True, device: typing.Any = None)
:canonical: elasticai.creator.nn.fixed_point.conv1d.layer.conv1d.Conv1d

Bases: {py:obj}`elasticai.creator.nn.design_creator_module.DesignCreatorModule`, {py:obj}`elasticai.creator.base_modules.conv1d.Conv1d`

````{py:method} create_design(name: str) -> elasticai.creator.nn.fixed_point.conv1d.design.Conv1dDesign
:canonical: elasticai.creator.nn.fixed_point.conv1d.layer.conv1d.Conv1d.create_design

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.conv1d.layer.conv1d.Conv1d.create_design
```

````

````{py:method} create_testbench(name: str, uut: elasticai.creator.nn.fixed_point.conv1d.design.Conv1dDesign) -> elasticai.creator.nn.fixed_point.conv1d.testbench.Conv1dTestbench
:canonical: elasticai.creator.nn.fixed_point.conv1d.layer.conv1d.Conv1d.create_testbench

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.conv1d.layer.conv1d.Conv1d.create_testbench
```

````

`````

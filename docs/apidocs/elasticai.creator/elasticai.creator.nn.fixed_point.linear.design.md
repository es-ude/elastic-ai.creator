# {py:mod}`elasticai.creator.nn.fixed_point.linear.design`

```{py:module} elasticai.creator.nn.fixed_point.linear.design
```

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.linear.design
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LinearDesign <elasticai.creator.nn.fixed_point.linear.design.LinearDesign>`
  -
````

### API

`````{py:class} LinearDesign(*, in_feature_num: int, out_feature_num: int, total_bits: int, frac_bits: int, weights: list[list[int]], bias: list[int], name: str, work_library_name: str = 'work', resource_option: str = 'auto')
:canonical: elasticai.creator.nn.fixed_point.linear.design.LinearDesign

Bases: {py:obj}`elasticai.creator.vhdl.design.design.Design`, {py:obj}`elasticai.creator.nn.fixed_point.linear.testbench.LinearDesignProtocol`

````{py:property} name
:canonical: elasticai.creator.nn.fixed_point.linear.design.LinearDesign.name

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.linear.design.LinearDesign.name
```

````

````{py:property} in_feature_num
:canonical: elasticai.creator.nn.fixed_point.linear.design.LinearDesign.in_feature_num
:type: int

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.linear.design.LinearDesign.in_feature_num
```

````

````{py:property} out_feature_num
:canonical: elasticai.creator.nn.fixed_point.linear.design.LinearDesign.out_feature_num
:type: int

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.linear.design.LinearDesign.out_feature_num
```

````

````{py:property} frac_width
:canonical: elasticai.creator.nn.fixed_point.linear.design.LinearDesign.frac_width
:type: int

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.linear.design.LinearDesign.frac_width
```

````

````{py:property} data_width
:canonical: elasticai.creator.nn.fixed_point.linear.design.LinearDesign.data_width
:type: int

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.linear.design.LinearDesign.data_width
```

````

````{py:property} port
:canonical: elasticai.creator.nn.fixed_point.linear.design.LinearDesign.port
:type: elasticai.creator.vhdl.design.ports.Port

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.linear.design.LinearDesign.port
```

````

````{py:method} save_to(destination: elasticai.creator.file_generation.savable.Path)
:canonical: elasticai.creator.nn.fixed_point.linear.design.LinearDesign.save_to

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.linear.design.LinearDesign.save_to
```

````

`````

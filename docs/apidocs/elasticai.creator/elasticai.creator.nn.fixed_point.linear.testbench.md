# {py:mod}`elasticai.creator.nn.fixed_point.linear.testbench`

```{py:module} elasticai.creator.nn.fixed_point.linear.testbench
```

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.linear.testbench
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LinearDesignProtocol <elasticai.creator.nn.fixed_point.linear.testbench.LinearDesignProtocol>`
  -
* - {py:obj}`LinearTestbench <elasticai.creator.nn.fixed_point.linear.testbench.LinearTestbench>`
  - ```{autodoc2-docstring} elasticai.creator.nn.fixed_point.linear.testbench.LinearTestbench
    :summary:
    ```
````

### API

`````{py:class} LinearDesignProtocol
:canonical: elasticai.creator.nn.fixed_point.linear.testbench.LinearDesignProtocol

Bases: {py:obj}`typing.Protocol`

````{py:property} name
:canonical: elasticai.creator.nn.fixed_point.linear.testbench.LinearDesignProtocol.name
:abstractmethod:
:type: str

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.linear.testbench.LinearDesignProtocol.name
```

````

````{py:property} port
:canonical: elasticai.creator.nn.fixed_point.linear.testbench.LinearDesignProtocol.port
:abstractmethod:
:type: elasticai.creator.vhdl.design.ports.Port

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.linear.testbench.LinearDesignProtocol.port
```

````

````{py:property} in_feature_num
:canonical: elasticai.creator.nn.fixed_point.linear.testbench.LinearDesignProtocol.in_feature_num
:abstractmethod:
:type: int

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.linear.testbench.LinearDesignProtocol.in_feature_num
```

````

````{py:property} out_feature_num
:canonical: elasticai.creator.nn.fixed_point.linear.testbench.LinearDesignProtocol.out_feature_num
:abstractmethod:
:type: int

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.linear.testbench.LinearDesignProtocol.out_feature_num
```

````

````{py:property} frac_width
:canonical: elasticai.creator.nn.fixed_point.linear.testbench.LinearDesignProtocol.frac_width
:abstractmethod:
:type: int

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.linear.testbench.LinearDesignProtocol.frac_width
```

````

````{py:property} data_width
:canonical: elasticai.creator.nn.fixed_point.linear.testbench.LinearDesignProtocol.data_width
:abstractmethod:
:type: int

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.linear.testbench.LinearDesignProtocol.data_width
```

````

`````

`````{py:class} LinearTestbench(name: str, uut: elasticai.creator.nn.fixed_point.linear.testbench.LinearDesignProtocol)
:canonical: elasticai.creator.nn.fixed_point.linear.testbench.LinearTestbench

Bases: {py:obj}`elasticai.creator.vhdl.simulated_layer.Testbench`

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.linear.testbench.LinearTestbench
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.linear.testbench.LinearTestbench.__init__
```

````{py:method} save_to(destination: elasticai.creator.file_generation.savable.Path)
:canonical: elasticai.creator.nn.fixed_point.linear.testbench.LinearTestbench.save_to

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.linear.testbench.LinearTestbench.save_to
```

````

````{py:property} name
:canonical: elasticai.creator.nn.fixed_point.linear.testbench.LinearTestbench.name
:type: str

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.linear.testbench.LinearTestbench.name
```

````

````{py:method} prepare_inputs(*inputs) -> list[dict]
:canonical: elasticai.creator.nn.fixed_point.linear.testbench.LinearTestbench.prepare_inputs

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.linear.testbench.LinearTestbench.prepare_inputs
```

````

````{py:method} parse_reported_content(content: list[str]) -> list[list[list[float]]]
:canonical: elasticai.creator.nn.fixed_point.linear.testbench.LinearTestbench.parse_reported_content

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.linear.testbench.LinearTestbench.parse_reported_content
```

````

`````

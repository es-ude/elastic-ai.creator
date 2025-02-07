# {py:mod}`elasticai.creator.nn.fixed_point.conv1d.design`

```{py:module} elasticai.creator.nn.fixed_point.conv1d.design
```

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.conv1d.design
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Conv1dDesign <elasticai.creator.nn.fixed_point.conv1d.design.Conv1dDesign>`
  -
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`generate_parameters_from_port <elasticai.creator.nn.fixed_point.conv1d.design.generate_parameters_from_port>`
  - ```{autodoc2-docstring} elasticai.creator.nn.fixed_point.conv1d.design.generate_parameters_from_port
    :summary:
    ```
````

### API

````{py:function} generate_parameters_from_port(port: elasticai.creator.vhdl.design.ports.Port) -> dict[str, str]
:canonical: elasticai.creator.nn.fixed_point.conv1d.design.generate_parameters_from_port

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.conv1d.design.generate_parameters_from_port
```
````

`````{py:class} Conv1dDesign(name: str, total_bits: int, frac_bits: int, in_channels: int, out_channels: int, signal_length: int, kernel_size: int, weights: list[list[list[int]]], bias: list[int])
:canonical: elasticai.creator.nn.fixed_point.conv1d.design.Conv1dDesign

Bases: {py:obj}`elasticai.creator.vhdl.design.design.Design`, {py:obj}`elasticai.creator.nn.fixed_point.conv1d.testbench.Conv1dDesignProtocol`

````{py:property} input_signal_length
:canonical: elasticai.creator.nn.fixed_point.conv1d.design.Conv1dDesign.input_signal_length
:type: int

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.conv1d.design.Conv1dDesign.input_signal_length
```

````

````{py:property} kernel_size
:canonical: elasticai.creator.nn.fixed_point.conv1d.design.Conv1dDesign.kernel_size
:type: int

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.conv1d.design.Conv1dDesign.kernel_size
```

````

````{py:property} port
:canonical: elasticai.creator.nn.fixed_point.conv1d.design.Conv1dDesign.port
:type: elasticai.creator.vhdl.design.ports.Port

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.conv1d.design.Conv1dDesign.port
```

````

````{py:property} in_channels
:canonical: elasticai.creator.nn.fixed_point.conv1d.design.Conv1dDesign.in_channels
:type: int

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.conv1d.design.Conv1dDesign.in_channels
```

````

````{py:property} out_channels
:canonical: elasticai.creator.nn.fixed_point.conv1d.design.Conv1dDesign.out_channels
:type: int

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.conv1d.design.Conv1dDesign.out_channels
```

````

````{py:method} save_to(destination: elasticai.creator.file_generation.savable.Path) -> None
:canonical: elasticai.creator.nn.fixed_point.conv1d.design.Conv1dDesign.save_to

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.conv1d.design.Conv1dDesign.save_to
```

````

`````

# {py:mod}`elasticai.creator.nn.fixed_point.conv1d.testbench`

```{py:module} elasticai.creator.nn.fixed_point.conv1d.testbench
```

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.conv1d.testbench
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Conv1dDesignProtocol <elasticai.creator.nn.fixed_point.conv1d.testbench.Conv1dDesignProtocol>`
  -
* - {py:obj}`Conv1dTestbench <elasticai.creator.nn.fixed_point.conv1d.testbench.Conv1dTestbench>`
  - ```{autodoc2-docstring} elasticai.creator.nn.fixed_point.conv1d.testbench.Conv1dTestbench
    :summary:
    ```
````

### API

`````{py:class} Conv1dDesignProtocol
:canonical: elasticai.creator.nn.fixed_point.conv1d.testbench.Conv1dDesignProtocol

Bases: {py:obj}`typing.Protocol`

````{py:property} name
:canonical: elasticai.creator.nn.fixed_point.conv1d.testbench.Conv1dDesignProtocol.name
:abstractmethod:
:type: str

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.conv1d.testbench.Conv1dDesignProtocol.name
```

````

````{py:property} input_signal_length
:canonical: elasticai.creator.nn.fixed_point.conv1d.testbench.Conv1dDesignProtocol.input_signal_length
:abstractmethod:
:type: int

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.conv1d.testbench.Conv1dDesignProtocol.input_signal_length
```

````

````{py:property} port
:canonical: elasticai.creator.nn.fixed_point.conv1d.testbench.Conv1dDesignProtocol.port
:abstractmethod:
:type: elasticai.creator.vhdl.design.ports.Port

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.conv1d.testbench.Conv1dDesignProtocol.port
```

````

````{py:property} kernel_size
:canonical: elasticai.creator.nn.fixed_point.conv1d.testbench.Conv1dDesignProtocol.kernel_size
:abstractmethod:
:type: int

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.conv1d.testbench.Conv1dDesignProtocol.kernel_size
```

````

````{py:property} in_channels
:canonical: elasticai.creator.nn.fixed_point.conv1d.testbench.Conv1dDesignProtocol.in_channels
:abstractmethod:
:type: int

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.conv1d.testbench.Conv1dDesignProtocol.in_channels
```

````

````{py:property} out_channels
:canonical: elasticai.creator.nn.fixed_point.conv1d.testbench.Conv1dDesignProtocol.out_channels
:abstractmethod:
:type: int

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.conv1d.testbench.Conv1dDesignProtocol.out_channels
```

````

`````

`````{py:class} Conv1dTestbench(name: str, uut: elasticai.creator.nn.fixed_point.conv1d.testbench.Conv1dDesignProtocol, fxp_params: elasticai.creator.nn.fixed_point.number_converter.FXPParams)
:canonical: elasticai.creator.nn.fixed_point.conv1d.testbench.Conv1dTestbench

Bases: {py:obj}`elasticai.creator.vhdl.simulated_layer.Testbench`

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.conv1d.testbench.Conv1dTestbench
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.conv1d.testbench.Conv1dTestbench.__init__
```

````{py:method} save_to(destination: elasticai.creator.file_generation.savable.Path)
:canonical: elasticai.creator.nn.fixed_point.conv1d.testbench.Conv1dTestbench.save_to

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.conv1d.testbench.Conv1dTestbench.save_to
```

````

````{py:property} name
:canonical: elasticai.creator.nn.fixed_point.conv1d.testbench.Conv1dTestbench.name
:type: str

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.conv1d.testbench.Conv1dTestbench.name
```

````

````{py:method} prepare_inputs(*inputs) -> list[dict]
:canonical: elasticai.creator.nn.fixed_point.conv1d.testbench.Conv1dTestbench.prepare_inputs

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.conv1d.testbench.Conv1dTestbench.prepare_inputs
```

````

````{py:method} parse_reported_content(content: list[str]) -> list[list[list[float]]]
:canonical: elasticai.creator.nn.fixed_point.conv1d.testbench.Conv1dTestbench.parse_reported_content

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.conv1d.testbench.Conv1dTestbench.parse_reported_content
```

````

`````

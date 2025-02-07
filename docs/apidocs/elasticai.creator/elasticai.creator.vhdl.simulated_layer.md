# {py:mod}`elasticai.creator.vhdl.simulated_layer`

```{py:module} elasticai.creator.vhdl.simulated_layer
```

```{autodoc2-docstring} elasticai.creator.vhdl.simulated_layer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Testbench <elasticai.creator.vhdl.simulated_layer.Testbench>`
  - ```{autodoc2-docstring} elasticai.creator.vhdl.simulated_layer.Testbench
    :summary:
    ```
* - {py:obj}`SimulatedLayer <elasticai.creator.vhdl.simulated_layer.SimulatedLayer>`
  - ```{autodoc2-docstring} elasticai.creator.vhdl.simulated_layer.SimulatedLayer
    :summary:
    ```
````

### API

`````{py:class} Testbench
:canonical: elasticai.creator.vhdl.simulated_layer.Testbench

```{autodoc2-docstring} elasticai.creator.vhdl.simulated_layer.Testbench
```

````{py:property} name
:canonical: elasticai.creator.vhdl.simulated_layer.Testbench.name
:abstractmethod:
:type: str

```{autodoc2-docstring} elasticai.creator.vhdl.simulated_layer.Testbench.name
```

````

````{py:method} save_to(destination: elasticai.creator.file_generation.savable.Path) -> None
:canonical: elasticai.creator.vhdl.simulated_layer.Testbench.save_to
:abstractmethod:

```{autodoc2-docstring} elasticai.creator.vhdl.simulated_layer.Testbench.save_to
```

````

````{py:method} prepare_inputs(*args: typing.Any, **kwargs: typing.Any) -> typing.Any
:canonical: elasticai.creator.vhdl.simulated_layer.Testbench.prepare_inputs
:abstractmethod:

```{autodoc2-docstring} elasticai.creator.vhdl.simulated_layer.Testbench.prepare_inputs
```

````

````{py:method} parse_reported_content(*args, **kwargs: typing.Any) -> typing.Any
:canonical: elasticai.creator.vhdl.simulated_layer.Testbench.parse_reported_content
:abstractmethod:

```{autodoc2-docstring} elasticai.creator.vhdl.simulated_layer.Testbench.parse_reported_content
```

````

`````

`````{py:class} SimulatedLayer(testbench: elasticai.creator.vhdl.simulated_layer.Testbench, simulator_constructor, working_dir: str | pathlib.Path)
:canonical: elasticai.creator.vhdl.simulated_layer.SimulatedLayer

```{autodoc2-docstring} elasticai.creator.vhdl.simulated_layer.SimulatedLayer
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.vhdl.simulated_layer.SimulatedLayer.__init__
```

````{py:method} __call__(inputs: typing.Any) -> typing.Any
:canonical: elasticai.creator.vhdl.simulated_layer.SimulatedLayer.__call__

```{autodoc2-docstring} elasticai.creator.vhdl.simulated_layer.SimulatedLayer.__call__
```

````

`````

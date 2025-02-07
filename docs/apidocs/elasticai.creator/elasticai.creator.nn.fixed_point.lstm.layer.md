# {py:mod}`elasticai.creator.nn.fixed_point.lstm.layer`

```{py:module} elasticai.creator.nn.fixed_point.lstm.layer
```

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.lstm.layer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LSTMNetwork <elasticai.creator.nn.fixed_point.lstm.layer.LSTMNetwork>`
  -
* - {py:obj}`FixedPointLSTMWithHardActivations <elasticai.creator.nn.fixed_point.lstm.layer.FixedPointLSTMWithHardActivations>`
  - ```{autodoc2-docstring} elasticai.creator.nn.fixed_point.lstm.layer.FixedPointLSTMWithHardActivations
    :summary:
    ```
````

### API

`````{py:class} LSTMNetwork(layers: list[torch.nn.Module])
:canonical: elasticai.creator.nn.fixed_point.lstm.layer.LSTMNetwork

Bases: {py:obj}`elasticai.creator.nn.design_creator_module.DesignCreatorModule`

````{py:method} create_design(name: str) -> elasticai.creator.nn.fixed_point.lstm.design.lstm.LSTMNetworkDesign
:canonical: elasticai.creator.nn.fixed_point.lstm.layer.LSTMNetwork.create_design

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.lstm.layer.LSTMNetwork.create_design
```

````

````{py:method} create_testbench(test_bench_name, uut: elasticai.creator.vhdl.design.design.Design) -> elasticai.creator.nn.fixed_point.lstm.design.testbench.LSTMTestBench
:canonical: elasticai.creator.nn.fixed_point.lstm.layer.LSTMNetwork.create_testbench

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.lstm.layer.LSTMNetwork.create_testbench
```

````

````{py:method} forward(x)
:canonical: elasticai.creator.nn.fixed_point.lstm.layer.LSTMNetwork.forward

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.lstm.layer.LSTMNetwork.forward
```

````

`````

`````{py:class} FixedPointLSTMWithHardActivations(total_bits: int, frac_bits: int, input_size: int, hidden_size: int, bias: bool)
:canonical: elasticai.creator.nn.fixed_point.lstm.layer.FixedPointLSTMWithHardActivations

Bases: {py:obj}`elasticai.creator.nn.design_creator_module.DesignCreatorModule`, {py:obj}`elasticai.creator.base_modules.lstm.LSTM`

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.lstm.layer.FixedPointLSTMWithHardActivations
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.lstm.layer.FixedPointLSTMWithHardActivations.__init__
```

````{py:property} fixed_point_config
:canonical: elasticai.creator.nn.fixed_point.lstm.layer.FixedPointLSTMWithHardActivations.fixed_point_config
:type: elasticai.creator.nn.fixed_point.two_complement_fixed_point_config.FixedPointConfig

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.lstm.layer.FixedPointLSTMWithHardActivations.fixed_point_config
```

````

````{py:method} create_design(name: str = 'lstm_cell') -> elasticai.creator.vhdl.design.design.Design
:canonical: elasticai.creator.nn.fixed_point.lstm.layer.FixedPointLSTMWithHardActivations.create_design

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.lstm.layer.FixedPointLSTMWithHardActivations.create_design
```

````

`````

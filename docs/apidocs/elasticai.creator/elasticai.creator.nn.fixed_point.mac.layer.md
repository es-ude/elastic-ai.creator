# {py:mod}`elasticai.creator.nn.fixed_point.mac.layer`

```{py:module} elasticai.creator.nn.fixed_point.mac.layer
```

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.mac.layer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MacLayer <elasticai.creator.nn.fixed_point.mac.layer.MacLayer>`
  - ```{autodoc2-docstring} elasticai.creator.nn.fixed_point.mac.layer.MacLayer
    :summary:
    ```
````

### API

`````{py:class} MacLayer(vector_width: int, fxp_params: elasticai.creator.nn.fixed_point.number_converter.FXPParams)
:canonical: elasticai.creator.nn.fixed_point.mac.layer.MacLayer

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.mac.layer.MacLayer
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.mac.layer.MacLayer.__init__
```

````{py:method} __call__(a, b)
:canonical: elasticai.creator.nn.fixed_point.mac.layer.MacLayer.__call__

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.mac.layer.MacLayer.__call__
```

````

````{py:method} create_design(name: str) -> elasticai.creator.file_generation.savable.Savable
:canonical: elasticai.creator.nn.fixed_point.mac.layer.MacLayer.create_design

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.mac.layer.MacLayer.create_design
```

````

````{py:method} create_testbench(name: str) -> elasticai.creator.nn.fixed_point.mac.mactestbench.MacTestBench
:canonical: elasticai.creator.nn.fixed_point.mac.layer.MacLayer.create_testbench

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.mac.layer.MacLayer.create_testbench
```

````

````{py:method} create_simulation(simulator, working_dir)
:canonical: elasticai.creator.nn.fixed_point.mac.layer.MacLayer.create_simulation

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.mac.layer.MacLayer.create_simulation
```

````

`````

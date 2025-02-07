# {py:mod}`elasticai.creator.vhdl.ghdl_simulation`

```{py:module} elasticai.creator.vhdl.ghdl_simulation
```

```{autodoc2-docstring} elasticai.creator.vhdl.ghdl_simulation
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GHDLSimulator <elasticai.creator.vhdl.ghdl_simulation.GHDLSimulator>`
  - ```{autodoc2-docstring} elasticai.creator.vhdl.ghdl_simulation.GHDLSimulator
    :summary:
    ```
````

### API

```{py:exception} SimulationError()
:canonical: elasticai.creator.vhdl.ghdl_simulation.SimulationError

Bases: {py:obj}`Exception`

```

`````{py:class} GHDLSimulator(workdir, top_design_name)
:canonical: elasticai.creator.vhdl.ghdl_simulation.GHDLSimulator

```{autodoc2-docstring} elasticai.creator.vhdl.ghdl_simulation.GHDLSimulator
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.vhdl.ghdl_simulation.GHDLSimulator.__init__
```

````{py:method} add_generic(**kwargs)
:canonical: elasticai.creator.vhdl.ghdl_simulation.GHDLSimulator.add_generic

```{autodoc2-docstring} elasticai.creator.vhdl.ghdl_simulation.GHDLSimulator.add_generic
```

````

````{py:method} initialize()
:canonical: elasticai.creator.vhdl.ghdl_simulation.GHDLSimulator.initialize

```{autodoc2-docstring} elasticai.creator.vhdl.ghdl_simulation.GHDLSimulator.initialize
```

````

````{py:method} run()
:canonical: elasticai.creator.vhdl.ghdl_simulation.GHDLSimulator.run

```{autodoc2-docstring} elasticai.creator.vhdl.ghdl_simulation.GHDLSimulator.run
```

````

````{py:method} getReportedContent() -> list[str]
:canonical: elasticai.creator.vhdl.ghdl_simulation.GHDLSimulator.getReportedContent

```{autodoc2-docstring} elasticai.creator.vhdl.ghdl_simulation.GHDLSimulator.getReportedContent
```

````

````{py:method} getFullReport() -> list[dict]
:canonical: elasticai.creator.vhdl.ghdl_simulation.GHDLSimulator.getFullReport

```{autodoc2-docstring} elasticai.creator.vhdl.ghdl_simulation.GHDLSimulator.getFullReport
```

````

````{py:method} getRawResult() -> str
:canonical: elasticai.creator.vhdl.ghdl_simulation.GHDLSimulator.getRawResult

```{autodoc2-docstring} elasticai.creator.vhdl.ghdl_simulation.GHDLSimulator.getRawResult
```

````

`````

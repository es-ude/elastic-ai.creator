# {py:mod}`elasticai.creator.vhdl.design.ports`

```{py:module} elasticai.creator.vhdl.design.ports
```

```{autodoc2-docstring} elasticai.creator.vhdl.design.ports
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Port <elasticai.creator.vhdl.design.ports.Port>`
  - ```{autodoc2-docstring} elasticai.creator.vhdl.design.ports.Port
    :summary:
    ```
````

### API

`````{py:class} Port(incoming: list[elasticai.creator.vhdl.design.signal.Signal], outgoing: list[elasticai.creator.vhdl.design.signal.Signal])
:canonical: elasticai.creator.vhdl.design.ports.Port

```{autodoc2-docstring} elasticai.creator.vhdl.design.ports.Port
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.vhdl.design.ports.Port.__init__
```

````{py:property} incoming
:canonical: elasticai.creator.vhdl.design.ports.Port.incoming
:type: list[elasticai.creator.vhdl.design.signal.Signal]

```{autodoc2-docstring} elasticai.creator.vhdl.design.ports.Port.incoming
```

````

````{py:property} outgoing
:canonical: elasticai.creator.vhdl.design.ports.Port.outgoing
:type: list[elasticai.creator.vhdl.design.signal.Signal]

```{autodoc2-docstring} elasticai.creator.vhdl.design.ports.Port.outgoing
```

````

````{py:property} signals
:canonical: elasticai.creator.vhdl.design.ports.Port.signals
:type: list[elasticai.creator.vhdl.design.signal.Signal]

```{autodoc2-docstring} elasticai.creator.vhdl.design.ports.Port.signals
```

````

````{py:property} signal_names
:canonical: elasticai.creator.vhdl.design.ports.Port.signal_names
:type: list[str]

```{autodoc2-docstring} elasticai.creator.vhdl.design.ports.Port.signal_names
```

````

````{py:method} __getitem__(item: str) -> elasticai.creator.vhdl.design.signal.Signal
:canonical: elasticai.creator.vhdl.design.ports.Port.__getitem__

```{autodoc2-docstring} elasticai.creator.vhdl.design.ports.Port.__getitem__
```

````

````{py:method} __contains__(item: str | elasticai.creator.vhdl.design.signal.Signal) -> bool
:canonical: elasticai.creator.vhdl.design.ports.Port.__contains__

```{autodoc2-docstring} elasticai.creator.vhdl.design.ports.Port.__contains__
```

````

````{py:method} __iter__() -> collections.abc.Iterator[elasticai.creator.vhdl.design.signal.Signal]
:canonical: elasticai.creator.vhdl.design.ports.Port.__iter__

```{autodoc2-docstring} elasticai.creator.vhdl.design.ports.Port.__iter__
```

````

`````

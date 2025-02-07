# {py:mod}`elasticai.creator.vhdl.auto_wire_protocols.autowiring`

```{py:module} elasticai.creator.vhdl.auto_wire_protocols.autowiring
```

```{autodoc2-docstring} elasticai.creator.vhdl.auto_wire_protocols.autowiring
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DataFlowNode <elasticai.creator.vhdl.auto_wire_protocols.autowiring.DataFlowNode>`
  - ```{autodoc2-docstring} elasticai.creator.vhdl.auto_wire_protocols.autowiring.DataFlowNode
    :summary:
    ```
* - {py:obj}`WiringProtocol <elasticai.creator.vhdl.auto_wire_protocols.autowiring.WiringProtocol>`
  - ```{autodoc2-docstring} elasticai.creator.vhdl.auto_wire_protocols.autowiring.WiringProtocol
    :summary:
    ```
* - {py:obj}`AutoWirer <elasticai.creator.vhdl.auto_wire_protocols.autowiring.AutoWirer>`
  - ```{autodoc2-docstring} elasticai.creator.vhdl.auto_wire_protocols.autowiring.AutoWirer
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BASIC_WIRING <elasticai.creator.vhdl.auto_wire_protocols.autowiring.BASIC_WIRING>`
  - ```{autodoc2-docstring} elasticai.creator.vhdl.auto_wire_protocols.autowiring.BASIC_WIRING
    :summary:
    ```
* - {py:obj}`T <elasticai.creator.vhdl.auto_wire_protocols.autowiring.T>`
  - ```{autodoc2-docstring} elasticai.creator.vhdl.auto_wire_protocols.autowiring.T
    :summary:
    ```
````

### API

`````{py:class} DataFlowNode
:canonical: elasticai.creator.vhdl.auto_wire_protocols.autowiring.DataFlowNode

```{autodoc2-docstring} elasticai.creator.vhdl.auto_wire_protocols.autowiring.DataFlowNode
```

````{py:attribute} sinks
:canonical: elasticai.creator.vhdl.auto_wire_protocols.autowiring.DataFlowNode.sinks
:type: tuple[str, ...]
:value: >
   None

```{autodoc2-docstring} elasticai.creator.vhdl.auto_wire_protocols.autowiring.DataFlowNode.sinks
```

````

````{py:attribute} sources
:canonical: elasticai.creator.vhdl.auto_wire_protocols.autowiring.DataFlowNode.sources
:type: tuple[str, ...]
:value: >
   None

```{autodoc2-docstring} elasticai.creator.vhdl.auto_wire_protocols.autowiring.DataFlowNode.sources
```

````

````{py:attribute} name
:canonical: elasticai.creator.vhdl.auto_wire_protocols.autowiring.DataFlowNode.name
:type: str
:value: >
   None

```{autodoc2-docstring} elasticai.creator.vhdl.auto_wire_protocols.autowiring.DataFlowNode.name
```

````

````{py:method} top(name: str) -> elasticai.creator.vhdl.auto_wire_protocols.autowiring.DataFlowNode
:canonical: elasticai.creator.vhdl.auto_wire_protocols.autowiring.DataFlowNode.top
:classmethod:

```{autodoc2-docstring} elasticai.creator.vhdl.auto_wire_protocols.autowiring.DataFlowNode.top
```

````

````{py:method} buffered(name: str) -> elasticai.creator.vhdl.auto_wire_protocols.autowiring.DataFlowNode
:canonical: elasticai.creator.vhdl.auto_wire_protocols.autowiring.DataFlowNode.buffered
:classmethod:

```{autodoc2-docstring} elasticai.creator.vhdl.auto_wire_protocols.autowiring.DataFlowNode.buffered
```

````

````{py:method} unbuffered(name: str) -> elasticai.creator.vhdl.auto_wire_protocols.autowiring.DataFlowNode
:canonical: elasticai.creator.vhdl.auto_wire_protocols.autowiring.DataFlowNode.unbuffered
:classmethod:

```{autodoc2-docstring} elasticai.creator.vhdl.auto_wire_protocols.autowiring.DataFlowNode.unbuffered
```

````

`````

`````{py:class} WiringProtocol
:canonical: elasticai.creator.vhdl.auto_wire_protocols.autowiring.WiringProtocol

```{autodoc2-docstring} elasticai.creator.vhdl.auto_wire_protocols.autowiring.WiringProtocol
```

````{py:attribute} up_sinks
:canonical: elasticai.creator.vhdl.auto_wire_protocols.autowiring.WiringProtocol.up_sinks
:type: dict[str, tuple[str, ...]]
:value: >
   None

```{autodoc2-docstring} elasticai.creator.vhdl.auto_wire_protocols.autowiring.WiringProtocol.up_sinks
```

````

````{py:attribute} down_sinks
:canonical: elasticai.creator.vhdl.auto_wire_protocols.autowiring.WiringProtocol.down_sinks
:type: dict[str, tuple[str, ...]]
:value: >
   None

```{autodoc2-docstring} elasticai.creator.vhdl.auto_wire_protocols.autowiring.WiringProtocol.down_sinks
```

````

`````

````{py:data} BASIC_WIRING
:canonical: elasticai.creator.vhdl.auto_wire_protocols.autowiring.BASIC_WIRING
:value: >
   'WiringProtocol(...)'

```{autodoc2-docstring} elasticai.creator.vhdl.auto_wire_protocols.autowiring.BASIC_WIRING
```

````

```{py:exception} AutoWiringProtocolViolation()
:canonical: elasticai.creator.vhdl.auto_wire_protocols.autowiring.AutoWiringProtocolViolation

Bases: {py:obj}`Exception`

```

````{py:data} T
:canonical: elasticai.creator.vhdl.auto_wire_protocols.autowiring.T
:value: >
   'TypeVar(...)'

```{autodoc2-docstring} elasticai.creator.vhdl.auto_wire_protocols.autowiring.T
```

````

`````{py:class} AutoWirer()
:canonical: elasticai.creator.vhdl.auto_wire_protocols.autowiring.AutoWirer

```{autodoc2-docstring} elasticai.creator.vhdl.auto_wire_protocols.autowiring.AutoWirer
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.vhdl.auto_wire_protocols.autowiring.AutoWirer.__init__
```

````{py:method} connections() -> dict[tuple[str, str], tuple[str, str]]
:canonical: elasticai.creator.vhdl.auto_wire_protocols.autowiring.AutoWirer.connections

```{autodoc2-docstring} elasticai.creator.vhdl.auto_wire_protocols.autowiring.AutoWirer.connections
```

````

````{py:method} wire(top: elasticai.creator.vhdl.auto_wire_protocols.autowiring.DataFlowNode, graph: collections.abc.Iterable[elasticai.creator.vhdl.auto_wire_protocols.autowiring.DataFlowNode])
:canonical: elasticai.creator.vhdl.auto_wire_protocols.autowiring.AutoWirer.wire

```{autodoc2-docstring} elasticai.creator.vhdl.auto_wire_protocols.autowiring.AutoWirer.wire
```

````

`````

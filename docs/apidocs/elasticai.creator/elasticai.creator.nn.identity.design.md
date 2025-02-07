# {py:mod}`elasticai.creator.nn.identity.design`

```{py:module} elasticai.creator.nn.identity.design
```

```{autodoc2-docstring} elasticai.creator.nn.identity.design
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BufferedIdentity <elasticai.creator.nn.identity.design.BufferedIdentity>`
  -
* - {py:obj}`BufferlessDesign <elasticai.creator.nn.identity.design.BufferlessDesign>`
  -
````

### API

`````{py:class} BufferedIdentity(name: str, num_input_features: int, num_input_bits: int)
:canonical: elasticai.creator.nn.identity.design.BufferedIdentity

Bases: {py:obj}`elasticai.creator.vhdl.design.design.Design`

````{py:property} port
:canonical: elasticai.creator.nn.identity.design.BufferedIdentity.port
:type: elasticai.creator.vhdl.design.ports.Port

```{autodoc2-docstring} elasticai.creator.nn.identity.design.BufferedIdentity.port
```

````

````{py:method} save_to(destination: elasticai.creator.file_generation.savable.Path) -> None
:canonical: elasticai.creator.nn.identity.design.BufferedIdentity.save_to

```{autodoc2-docstring} elasticai.creator.nn.identity.design.BufferedIdentity.save_to
```

````

`````

`````{py:class} BufferlessDesign(name: str, num_input_bits: int)
:canonical: elasticai.creator.nn.identity.design.BufferlessDesign

Bases: {py:obj}`elasticai.creator.vhdl.design.design.Design`

````{py:property} port
:canonical: elasticai.creator.nn.identity.design.BufferlessDesign.port
:type: elasticai.creator.vhdl.design.ports.Port

```{autodoc2-docstring} elasticai.creator.nn.identity.design.BufferlessDesign.port
```

````

````{py:method} save_to(destination: elasticai.creator.file_generation.savable.Path) -> None
:canonical: elasticai.creator.nn.identity.design.BufferlessDesign.save_to

```{autodoc2-docstring} elasticai.creator.nn.identity.design.BufferlessDesign.save_to
```

````

`````

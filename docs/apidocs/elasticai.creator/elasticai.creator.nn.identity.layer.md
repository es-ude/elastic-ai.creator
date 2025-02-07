# {py:mod}`elasticai.creator.nn.identity.layer`

```{py:module} elasticai.creator.nn.identity.layer
```

```{autodoc2-docstring} elasticai.creator.nn.identity.layer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BufferedIdentity <elasticai.creator.nn.identity.layer.BufferedIdentity>`
  -
* - {py:obj}`BufferlessIdentity <elasticai.creator.nn.identity.layer.BufferlessIdentity>`
  -
````

### API

`````{py:class} BufferedIdentity(num_input_features: int, total_bits: int)
:canonical: elasticai.creator.nn.identity.layer.BufferedIdentity

Bases: {py:obj}`elasticai.creator.nn.design_creator_module.DesignCreatorModule`, {py:obj}`elasticai.creator.base_modules.identity.Identity`

````{py:method} create_design(name: str) -> elasticai.creator.vhdl.design.design.Design
:canonical: elasticai.creator.nn.identity.layer.BufferedIdentity.create_design

```{autodoc2-docstring} elasticai.creator.nn.identity.layer.BufferedIdentity.create_design
```

````

`````

`````{py:class} BufferlessIdentity(total_bits: int)
:canonical: elasticai.creator.nn.identity.layer.BufferlessIdentity

Bases: {py:obj}`elasticai.creator.nn.design_creator_module.DesignCreatorModule`, {py:obj}`elasticai.creator.base_modules.identity.Identity`

````{py:method} create_design(name: str) -> elasticai.creator.vhdl.design.design.Design
:canonical: elasticai.creator.nn.identity.layer.BufferlessIdentity.create_design

```{autodoc2-docstring} elasticai.creator.nn.identity.layer.BufferlessIdentity.create_design
```

````

`````

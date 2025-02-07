# {py:mod}`elasticai.creator.ir.ir_data`

```{py:module} elasticai.creator.ir.ir_data
```

```{autodoc2-docstring} elasticai.creator.ir.ir_data
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`IrData <elasticai.creator.ir.ir_data.IrData>`
  - ```{autodoc2-docstring} elasticai.creator.ir.ir_data.IrData
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ir_data_class <elasticai.creator.ir.ir_data.ir_data_class>`
  - ```{autodoc2-docstring} elasticai.creator.ir.ir_data.ir_data_class
    :summary:
    ```
````

### API

`````{py:class} IrData(data: dict[str, elasticai.creator.ir.attribute.Attribute])
:canonical: elasticai.creator.ir.ir_data.IrData

```{autodoc2-docstring} elasticai.creator.ir.ir_data.IrData
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.ir.ir_data.IrData.__init__
```

````{py:attribute} attributes
:canonical: elasticai.creator.ir.ir_data.IrData.attributes
:type: elasticai.creator.ir.attributes_descriptor.AttributesDescriptor
:value: >
   'AttributesDescriptor(...)'

```{autodoc2-docstring} elasticai.creator.ir.ir_data.IrData.attributes
```

````

````{py:method} get_missing_required_fields() -> dict[str, type]
:canonical: elasticai.creator.ir.ir_data.IrData.get_missing_required_fields

```{autodoc2-docstring} elasticai.creator.ir.ir_data.IrData.get_missing_required_fields
```

````

````{py:method} __repr__() -> str
:canonical: elasticai.creator.ir.ir_data.IrData.__repr__

````

````{py:method} __eq__(o: object) -> bool
:canonical: elasticai.creator.ir.ir_data.IrData.__eq__

````

`````

````{py:function} ir_data_class(cls) -> collections.abc.Callable[[dict[str, elasticai.creator.ir.attribute.Attribute]], elasticai.creator.ir.ir_data.IrData]
:canonical: elasticai.creator.ir.ir_data.ir_data_class

```{autodoc2-docstring} elasticai.creator.ir.ir_data.ir_data_class
```
````

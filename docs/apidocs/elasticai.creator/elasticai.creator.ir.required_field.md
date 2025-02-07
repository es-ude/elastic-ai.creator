# {py:mod}`elasticai.creator.ir.required_field`

```{py:module} elasticai.creator.ir.required_field
```

```{autodoc2-docstring} elasticai.creator.ir.required_field
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RequiredField <elasticai.creator.ir.required_field.RequiredField>`
  - ```{autodoc2-docstring} elasticai.creator.ir.required_field.RequiredField
    :summary:
    ```
* - {py:obj}`SimpleRequiredField <elasticai.creator.ir.required_field.SimpleRequiredField>`
  -
* - {py:obj}`ReadOnlyField <elasticai.creator.ir.required_field.ReadOnlyField>`
  -
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`is_required_field <elasticai.creator.ir.required_field.is_required_field>`
  - ```{autodoc2-docstring} elasticai.creator.ir.required_field.is_required_field
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`StoredT <elasticai.creator.ir.required_field.StoredT>`
  - ```{autodoc2-docstring} elasticai.creator.ir.required_field.StoredT
    :summary:
    ```
* - {py:obj}`VisibleT <elasticai.creator.ir.required_field.VisibleT>`
  - ```{autodoc2-docstring} elasticai.creator.ir.required_field.VisibleT
    :summary:
    ```
````

### API

````{py:data} StoredT
:canonical: elasticai.creator.ir.required_field.StoredT
:value: >
   'TypeVar(...)'

```{autodoc2-docstring} elasticai.creator.ir.required_field.StoredT
```

````

````{py:data} VisibleT
:canonical: elasticai.creator.ir.required_field.VisibleT
:value: >
   'TypeVar(...)'

```{autodoc2-docstring} elasticai.creator.ir.required_field.VisibleT
```

````

`````{py:class} RequiredField(set_convert: collections.abc.Callable[[elasticai.creator.ir.required_field.VisibleT], elasticai.creator.ir.required_field.StoredT], get_convert: collections.abc.Callable[[elasticai.creator.ir.required_field.StoredT], elasticai.creator.ir.required_field.VisibleT])
:canonical: elasticai.creator.ir.required_field.RequiredField

Bases: {py:obj}`typing.Generic`\[{py:obj}`elasticai.creator.ir.required_field.StoredT`\, {py:obj}`elasticai.creator.ir.required_field.VisibleT`\]

```{autodoc2-docstring} elasticai.creator.ir.required_field.RequiredField
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.ir.required_field.RequiredField.__init__
```

````{py:attribute} __slots__
:canonical: elasticai.creator.ir.required_field.RequiredField.__slots__
:value: >
   ('set_convert', 'get_convert', 'name')

```{autodoc2-docstring} elasticai.creator.ir.required_field.RequiredField.__slots__
```

````

````{py:method} __set_name__(owner: type[elasticai.creator.ir._has_data.HasData], name: str) -> None
:canonical: elasticai.creator.ir.required_field.RequiredField.__set_name__

```{autodoc2-docstring} elasticai.creator.ir.required_field.RequiredField.__set_name__
```

````

````{py:method} __get__(instance: elasticai.creator.ir._has_data.HasData, owner=None) -> elasticai.creator.ir.required_field.VisibleT
:canonical: elasticai.creator.ir.required_field.RequiredField.__get__

```{autodoc2-docstring} elasticai.creator.ir.required_field.RequiredField.__get__
```

````

````{py:method} __set__(instance: elasticai.creator.ir._has_data.HasData, value: elasticai.creator.ir.required_field.VisibleT) -> None
:canonical: elasticai.creator.ir.required_field.RequiredField.__set__

```{autodoc2-docstring} elasticai.creator.ir.required_field.RequiredField.__set__
```

````

`````

`````{py:class} SimpleRequiredField()
:canonical: elasticai.creator.ir.required_field.SimpleRequiredField

Bases: {py:obj}`elasticai.creator.ir.required_field.RequiredField`\[{py:obj}`elasticai.creator.ir.required_field.StoredT`\, {py:obj}`elasticai.creator.ir.required_field.StoredT`\]

````{py:attribute} slots
:canonical: elasticai.creator.ir.required_field.SimpleRequiredField.slots
:value: >
   ('get_convert', 'set_convert', 'name')

```{autodoc2-docstring} elasticai.creator.ir.required_field.SimpleRequiredField.slots
```

````

`````

`````{py:class} ReadOnlyField(get_convert: collections.abc.Callable[[elasticai.creator.ir.required_field.StoredT], elasticai.creator.ir.required_field.VisibleT])
:canonical: elasticai.creator.ir.required_field.ReadOnlyField

Bases: {py:obj}`typing.Generic`\[{py:obj}`elasticai.creator.ir.required_field.StoredT`\, {py:obj}`elasticai.creator.ir.required_field.VisibleT`\]

````{py:attribute} slots
:canonical: elasticai.creator.ir.required_field.ReadOnlyField.slots
:value: >
   ('get_convert',)

```{autodoc2-docstring} elasticai.creator.ir.required_field.ReadOnlyField.slots
```

````

````{py:method} __set_name__(owner: type[elasticai.creator.ir._has_data.HasData], name: str) -> None
:canonical: elasticai.creator.ir.required_field.ReadOnlyField.__set_name__

```{autodoc2-docstring} elasticai.creator.ir.required_field.ReadOnlyField.__set_name__
```

````

````{py:method} __get__(instance: elasticai.creator.ir._has_data.HasData, owner: type[elasticai.creator.ir._has_data.HasData] | None = None) -> elasticai.creator.ir.required_field.VisibleT
:canonical: elasticai.creator.ir.required_field.ReadOnlyField.__get__

```{autodoc2-docstring} elasticai.creator.ir.required_field.ReadOnlyField.__get__
```

````

`````

````{py:function} is_required_field(o: object) -> typing_extensions.TypeIs[elasticai.creator.ir.required_field.RequiredField | elasticai.creator.ir.required_field.ReadOnlyField]
:canonical: elasticai.creator.ir.required_field.is_required_field

```{autodoc2-docstring} elasticai.creator.ir.required_field.is_required_field
```
````

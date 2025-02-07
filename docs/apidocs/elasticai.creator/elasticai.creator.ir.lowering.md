# {py:mod}`elasticai.creator.ir.lowering`

```{py:module} elasticai.creator.ir.lowering
```

```{autodoc2-docstring} elasticai.creator.ir.lowering
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Lowerable <elasticai.creator.ir.lowering.Lowerable>`
  -
* - {py:obj}`LoweringPass <elasticai.creator.ir.lowering.LoweringPass>`
  -
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`return_as_iterable <elasticai.creator.ir.lowering.return_as_iterable>`
  - ```{autodoc2-docstring} elasticai.creator.ir.lowering.return_as_iterable
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Tin <elasticai.creator.ir.lowering.Tin>`
  - ```{autodoc2-docstring} elasticai.creator.ir.lowering.Tin
    :summary:
    ```
* - {py:obj}`Tout <elasticai.creator.ir.lowering.Tout>`
  - ```{autodoc2-docstring} elasticai.creator.ir.lowering.Tout
    :summary:
    ```
* - {py:obj}`P <elasticai.creator.ir.lowering.P>`
  - ```{autodoc2-docstring} elasticai.creator.ir.lowering.P
    :summary:
    ```
````

### API

`````{py:class} Lowerable
:canonical: elasticai.creator.ir.lowering.Lowerable

Bases: {py:obj}`typing.Protocol`

````{py:property} type
:canonical: elasticai.creator.ir.lowering.Lowerable.type
:abstractmethod:
:type: str

```{autodoc2-docstring} elasticai.creator.ir.lowering.Lowerable.type
```

````

`````

````{py:data} Tin
:canonical: elasticai.creator.ir.lowering.Tin
:value: >
   'TypeVar(...)'

```{autodoc2-docstring} elasticai.creator.ir.lowering.Tin
```

````

````{py:data} Tout
:canonical: elasticai.creator.ir.lowering.Tout
:value: >
   'TypeVar(...)'

```{autodoc2-docstring} elasticai.creator.ir.lowering.Tout
```

````

`````{py:class} LoweringPass()
:canonical: elasticai.creator.ir.lowering.LoweringPass

Bases: {py:obj}`typing.Generic`\[{py:obj}`elasticai.creator.ir.lowering.Tin`\, {py:obj}`elasticai.creator.ir.lowering.Tout`\]

````{py:attribute} register
:canonical: elasticai.creator.ir.lowering.LoweringPass.register
:type: elasticai.creator.function_utils.RegisterDescriptor[elasticai.creator.ir.lowering.Tin, elasticai.creator.ir.lowering.Tout]
:value: >
   'RegisterDescriptor(...)'

```{autodoc2-docstring} elasticai.creator.ir.lowering.LoweringPass.register
```

````

````{py:attribute} register_iterable
:canonical: elasticai.creator.ir.lowering.LoweringPass.register_iterable
:type: elasticai.creator.function_utils.RegisterDescriptor[elasticai.creator.ir.lowering.Tin, collections.abc.Iterable[elasticai.creator.ir.lowering.Tout]]
:value: >
   'RegisterDescriptor(...)'

```{autodoc2-docstring} elasticai.creator.ir.lowering.LoweringPass.register_iterable
```

````

````{py:method} __call__(args: collections.abc.Iterable[elasticai.creator.ir.lowering.Tin]) -> collections.abc.Iterator[elasticai.creator.ir.lowering.Tout]
:canonical: elasticai.creator.ir.lowering.LoweringPass.__call__

```{autodoc2-docstring} elasticai.creator.ir.lowering.LoweringPass.__call__
```

````

`````

````{py:data} P
:canonical: elasticai.creator.ir.lowering.P
:value: >
   'ParamSpec(...)'

```{autodoc2-docstring} elasticai.creator.ir.lowering.P
```

````

````{py:function} return_as_iterable(fn: collections.abc.Callable[elasticai.creator.ir.lowering.P, elasticai.creator.ir.lowering.Tout]) -> collections.abc.Callable[elasticai.creator.ir.lowering.P, collections.abc.Iterable[elasticai.creator.ir.lowering.Tout]]
:canonical: elasticai.creator.ir.lowering.return_as_iterable

```{autodoc2-docstring} elasticai.creator.ir.lowering.return_as_iterable
```
````

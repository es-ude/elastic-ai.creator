# {py:mod}`elasticai.creator.ir2torch.ir2torch`

```{py:module} elasticai.creator.ir2torch.ir2torch
```

```{autodoc2-docstring} elasticai.creator.ir2torch.ir2torch
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Ir2Torch <elasticai.creator.ir2torch.ir2torch.Ir2Torch>`
  - ```{autodoc2-docstring} elasticai.creator.ir2torch.ir2torch.Ir2Torch
    :summary:
    ```
````

### API

`````{py:class} Ir2Torch()
:canonical: elasticai.creator.ir2torch.ir2torch.Ir2Torch

```{autodoc2-docstring} elasticai.creator.ir2torch.ir2torch.Ir2Torch
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.ir2torch.ir2torch.Ir2Torch.__init__
```

````{py:method} convert(ir: dict[str, elasticai.creator.torch2ir.Implementation]) -> torch.nn.Module
:canonical: elasticai.creator.ir2torch.ir2torch.Ir2Torch.convert

```{autodoc2-docstring} elasticai.creator.ir2torch.ir2torch.Ir2Torch.convert
```

````

````{py:method} register_type_handlers(handlers: collections.abc.Iterable[collections.abc.Callable[[elasticai.creator.torch2ir.Implementation], torch.nn.Module]]) -> None
:canonical: elasticai.creator.ir2torch.ir2torch.Ir2Torch.register_type_handlers

```{autodoc2-docstring} elasticai.creator.ir2torch.ir2torch.Ir2Torch.register_type_handlers
```

````

````{py:method} get_default_converter() -> elasticai.creator.ir2torch.ir2torch.Ir2Torch
:canonical: elasticai.creator.ir2torch.ir2torch.Ir2Torch.get_default_converter
:classmethod:

```{autodoc2-docstring} elasticai.creator.ir2torch.ir2torch.Ir2Torch.get_default_converter
```

````

`````

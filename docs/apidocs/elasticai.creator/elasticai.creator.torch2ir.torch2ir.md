# {py:mod}`elasticai.creator.torch2ir.torch2ir`

```{py:module} elasticai.creator.torch2ir.torch2ir
```

```{autodoc2-docstring} elasticai.creator.torch2ir.torch2ir
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Torch2Ir <elasticai.creator.torch2ir.torch2ir.Torch2Ir>`
  - ```{autodoc2-docstring} elasticai.creator.torch2ir.torch2ir.Torch2Ir
    :summary:
    ```
````

### API

```{py:exception} LoweringError(message: str)
:canonical: elasticai.creator.torch2ir.torch2ir.LoweringError

Bases: {py:obj}`Exception`

```

`````{py:class} Torch2Ir(tracer: torch.fx.Tracer = _DefaultTracer())
:canonical: elasticai.creator.torch2ir.torch2ir.Torch2Ir

```{autodoc2-docstring} elasticai.creator.torch2ir.torch2ir.Torch2Ir
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.torch2ir.torch2ir.Torch2Ir.__init__
```

````{py:method} register_module_handler(module_type: str, handler: collections.abc.Callable[[torch.nn.Module], dict]) -> None
:canonical: elasticai.creator.torch2ir.torch2ir.Torch2Ir.register_module_handler

```{autodoc2-docstring} elasticai.creator.torch2ir.torch2ir.Torch2Ir.register_module_handler
```

````

````{py:method} register_module_handlers(handlers: collections.abc.Iterable[collections.abc.Callable[[torch.nn.Module], dict]]) -> elasticai.creator.torch2ir.torch2ir.Torch2Ir
:canonical: elasticai.creator.torch2ir.torch2ir.Torch2Ir.register_module_handlers

```{autodoc2-docstring} elasticai.creator.torch2ir.torch2ir.Torch2Ir.register_module_handlers
```

````

````{py:method} convert(model: torch.nn.Module) -> dict[str, elasticai.creator.torch2ir.core.Implementation]
:canonical: elasticai.creator.torch2ir.torch2ir.Torch2Ir.convert

```{autodoc2-docstring} elasticai.creator.torch2ir.torch2ir.Torch2Ir.convert
```

````

````{py:method} get_default_converter() -> elasticai.creator.torch2ir.torch2ir.Torch2Ir
:canonical: elasticai.creator.torch2ir.torch2ir.Torch2Ir.get_default_converter
:classmethod:

```{autodoc2-docstring} elasticai.creator.torch2ir.torch2ir.Torch2Ir.get_default_converter
```

````

`````

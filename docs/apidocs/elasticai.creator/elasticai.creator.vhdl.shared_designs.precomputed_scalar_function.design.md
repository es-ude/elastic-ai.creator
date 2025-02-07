# {py:mod}`elasticai.creator.vhdl.shared_designs.precomputed_scalar_function.design`

```{py:module} elasticai.creator.vhdl.shared_designs.precomputed_scalar_function.design
```

```{autodoc2-docstring} elasticai.creator.vhdl.shared_designs.precomputed_scalar_function.design
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PrecomputedScalarFunction <elasticai.creator.vhdl.shared_designs.precomputed_scalar_function.design.PrecomputedScalarFunction>`
  -
````

### API

`````{py:class} PrecomputedScalarFunction(name: str, input_width: int, output_width: int, function: collections.abc.Callable[[int], int], inputs: list[int])
:canonical: elasticai.creator.vhdl.shared_designs.precomputed_scalar_function.design.PrecomputedScalarFunction

Bases: {py:obj}`elasticai.creator.vhdl.design.design.Design`

````{py:property} port
:canonical: elasticai.creator.vhdl.shared_designs.precomputed_scalar_function.design.PrecomputedScalarFunction.port
:type: elasticai.creator.vhdl.design.ports.Port

```{autodoc2-docstring} elasticai.creator.vhdl.shared_designs.precomputed_scalar_function.design.PrecomputedScalarFunction.port
```

````

````{py:method} save_to(destination: elasticai.creator.file_generation.savable.Path) -> None
:canonical: elasticai.creator.vhdl.shared_designs.precomputed_scalar_function.design.PrecomputedScalarFunction.save_to

```{autodoc2-docstring} elasticai.creator.vhdl.shared_designs.precomputed_scalar_function.design.PrecomputedScalarFunction.save_to
```

````

`````

# {py:mod}`elasticai.creator.nn.fixed_point.precomputed.precomputed_module`

```{py:module} elasticai.creator.nn.fixed_point.precomputed.precomputed_module
```

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.precomputed.precomputed_module
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PrecomputedModule <elasticai.creator.nn.fixed_point.precomputed.precomputed_module.PrecomputedModule>`
  -
````

### API

`````{py:class} PrecomputedModule(base_module: torch.nn.Module, total_bits: int, frac_bits: int, num_steps: int, sampling_intervall: tuple[float, float])
:canonical: elasticai.creator.nn.fixed_point.precomputed.precomputed_module.PrecomputedModule

Bases: {py:obj}`elasticai.creator.nn.design_creator_module.DesignCreatorModule`

````{py:method} forward(x: torch.Tensor) -> torch.Tensor
:canonical: elasticai.creator.nn.fixed_point.precomputed.precomputed_module.PrecomputedModule.forward

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.precomputed.precomputed_module.PrecomputedModule.forward
```

````

````{py:method} create_design(name: str) -> elasticai.creator.vhdl.shared_designs.precomputed_scalar_function.PrecomputedScalarFunction
:canonical: elasticai.creator.nn.fixed_point.precomputed.precomputed_module.PrecomputedModule.create_design

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.precomputed.precomputed_module.PrecomputedModule.create_design
```

````

`````

# {py:mod}`elasticai.creator.vhdl.system_integrations.skeleton.skeleton`

```{py:module} elasticai.creator.vhdl.system_integrations.skeleton.skeleton
```

```{autodoc2-docstring} elasticai.creator.vhdl.system_integrations.skeleton.skeleton
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Skeleton <elasticai.creator.vhdl.system_integrations.skeleton.skeleton.Skeleton>`
  - ```{autodoc2-docstring} elasticai.creator.vhdl.system_integrations.skeleton.skeleton.Skeleton
    :summary:
    ```
* - {py:obj}`LSTMSkeleton <elasticai.creator.vhdl.system_integrations.skeleton.skeleton.LSTMSkeleton>`
  - ```{autodoc2-docstring} elasticai.creator.vhdl.system_integrations.skeleton.skeleton.LSTMSkeleton
    :summary:
    ```
* - {py:obj}`EchoSkeletonV2 <elasticai.creator.vhdl.system_integrations.skeleton.skeleton.EchoSkeletonV2>`
  - ```{autodoc2-docstring} elasticai.creator.vhdl.system_integrations.skeleton.skeleton.EchoSkeletonV2
    :summary:
    ```
````

### API

`````{py:class} Skeleton(x_num_values: int, y_num_values: int, network_name: str, port: elasticai.creator.vhdl.design.ports.Port, id: list[int] | int, skeleton_version: str = 'v1')
:canonical: elasticai.creator.vhdl.system_integrations.skeleton.skeleton.Skeleton

```{autodoc2-docstring} elasticai.creator.vhdl.system_integrations.skeleton.skeleton.Skeleton
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.vhdl.system_integrations.skeleton.skeleton.Skeleton.__init__
```

````{py:method} save_to(destination: elasticai.creator.file_generation.savable.Path)
:canonical: elasticai.creator.vhdl.system_integrations.skeleton.skeleton.Skeleton.save_to

```{autodoc2-docstring} elasticai.creator.vhdl.system_integrations.skeleton.skeleton.Skeleton.save_to
```

````

`````

`````{py:class} LSTMSkeleton(network_name: str)
:canonical: elasticai.creator.vhdl.system_integrations.skeleton.skeleton.LSTMSkeleton

```{autodoc2-docstring} elasticai.creator.vhdl.system_integrations.skeleton.skeleton.LSTMSkeleton
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.vhdl.system_integrations.skeleton.skeleton.LSTMSkeleton.__init__
```

````{py:method} save_to(destination: elasticai.creator.file_generation.savable.Path)
:canonical: elasticai.creator.vhdl.system_integrations.skeleton.skeleton.LSTMSkeleton.save_to

```{autodoc2-docstring} elasticai.creator.vhdl.system_integrations.skeleton.skeleton.LSTMSkeleton.save_to
```

````

`````

`````{py:class} EchoSkeletonV2(num_values: int, bitwidth: int)
:canonical: elasticai.creator.vhdl.system_integrations.skeleton.skeleton.EchoSkeletonV2

```{autodoc2-docstring} elasticai.creator.vhdl.system_integrations.skeleton.skeleton.EchoSkeletonV2
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.vhdl.system_integrations.skeleton.skeleton.EchoSkeletonV2.__init__
```

````{py:method} save_to(destination: elasticai.creator.file_generation.savable.Path)
:canonical: elasticai.creator.vhdl.system_integrations.skeleton.skeleton.EchoSkeletonV2.save_to

```{autodoc2-docstring} elasticai.creator.vhdl.system_integrations.skeleton.skeleton.EchoSkeletonV2.save_to
```

````

`````

# {py:mod}`elasticai.creator.vhdl.system_integrations.firmware_env5`

```{py:module} elasticai.creator.vhdl.system_integrations.firmware_env5
```

```{autodoc2-docstring} elasticai.creator.vhdl.system_integrations.firmware_env5
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SkeletonType <elasticai.creator.vhdl.system_integrations.firmware_env5.SkeletonType>`
  -
* - {py:obj}`FirmwareENv5 <elasticai.creator.vhdl.system_integrations.firmware_env5.FirmwareENv5>`
  - ```{autodoc2-docstring} elasticai.creator.vhdl.system_integrations.firmware_env5.FirmwareENv5
    :summary:
    ```
* - {py:obj}`LSTMFirmwareENv5 <elasticai.creator.vhdl.system_integrations.firmware_env5.LSTMFirmwareENv5>`
  - ```{autodoc2-docstring} elasticai.creator.vhdl.system_integrations.firmware_env5.LSTMFirmwareENv5
    :summary:
    ```
````

### API

`````{py:class} SkeletonType
:canonical: elasticai.creator.vhdl.system_integrations.firmware_env5.SkeletonType

Bases: {py:obj}`typing.Protocol`

````{py:method} save_to(destination: elasticai.creator.file_generation.savable.Path) -> None
:canonical: elasticai.creator.vhdl.system_integrations.firmware_env5.SkeletonType.save_to

```{autodoc2-docstring} elasticai.creator.vhdl.system_integrations.firmware_env5.SkeletonType.save_to
```

````

`````

````{py:class} FirmwareENv5(network: elasticai.creator.vhdl.design.design.Design, x_num_values: int, y_num_values: int, id: list[int] | int, skeleton_version: str = 'v1')
:canonical: elasticai.creator.vhdl.system_integrations.firmware_env5.FirmwareENv5

Bases: {py:obj}`elasticai.creator.vhdl.system_integrations.firmware_env5._FirmwareENv5Base`

```{autodoc2-docstring} elasticai.creator.vhdl.system_integrations.firmware_env5.FirmwareENv5
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.vhdl.system_integrations.firmware_env5.FirmwareENv5.__init__
```

````

````{py:class} LSTMFirmwareENv5(network: elasticai.creator.vhdl.design.design.Design)
:canonical: elasticai.creator.vhdl.system_integrations.firmware_env5.LSTMFirmwareENv5

Bases: {py:obj}`elasticai.creator.vhdl.system_integrations.firmware_env5._FirmwareENv5Base`

```{autodoc2-docstring} elasticai.creator.vhdl.system_integrations.firmware_env5.LSTMFirmwareENv5
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.vhdl.system_integrations.firmware_env5.LSTMFirmwareENv5.__init__
```

````

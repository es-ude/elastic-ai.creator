# {py:mod}`elasticai.creator.vhdl.auto_wire_protocols.port_definitions`

```{py:module} elasticai.creator.vhdl.auto_wire_protocols.port_definitions
```

```{autodoc2-docstring} elasticai.creator.vhdl.auto_wire_protocols.port_definitions
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`create_port <elasticai.creator.vhdl.auto_wire_protocols.port_definitions.create_port>`
  - ```{autodoc2-docstring} elasticai.creator.vhdl.auto_wire_protocols.port_definitions.create_port
    :summary:
    ```
* - {py:obj}`create_port_for_bufferless_design <elasticai.creator.vhdl.auto_wire_protocols.port_definitions.create_port_for_bufferless_design>`
  - ```{autodoc2-docstring} elasticai.creator.vhdl.auto_wire_protocols.port_definitions.create_port_for_bufferless_design
    :summary:
    ```
* - {py:obj}`create_port_for_buffered_design <elasticai.creator.vhdl.auto_wire_protocols.port_definitions.create_port_for_buffered_design>`
  - ```{autodoc2-docstring} elasticai.creator.vhdl.auto_wire_protocols.port_definitions.create_port_for_buffered_design
    :summary:
    ```
* - {py:obj}`port_definition_template_for_buffered_design <elasticai.creator.vhdl.auto_wire_protocols.port_definitions.port_definition_template_for_buffered_design>`
  - ```{autodoc2-docstring} elasticai.creator.vhdl.auto_wire_protocols.port_definitions.port_definition_template_for_buffered_design
    :summary:
    ```
* - {py:obj}`port_definition_template_for_bufferless_design <elasticai.creator.vhdl.auto_wire_protocols.port_definitions.port_definition_template_for_bufferless_design>`
  - ```{autodoc2-docstring} elasticai.creator.vhdl.auto_wire_protocols.port_definitions.port_definition_template_for_bufferless_design
    :summary:
    ```
````

### API

````{py:function} create_port(x_width: int, y_width: int, *, x_count: int = 0, y_count: int = 0, x_address_width: typing.Optional[int] = None, y_address_width: typing.Optional[int] = None) -> elasticai.creator.vhdl.design.ports.Port
:canonical: elasticai.creator.vhdl.auto_wire_protocols.port_definitions.create_port

```{autodoc2-docstring} elasticai.creator.vhdl.auto_wire_protocols.port_definitions.create_port
```
````

````{py:function} create_port_for_bufferless_design(x_width: int, y_width: int) -> elasticai.creator.vhdl.design.ports.Port
:canonical: elasticai.creator.vhdl.auto_wire_protocols.port_definitions.create_port_for_bufferless_design

```{autodoc2-docstring} elasticai.creator.vhdl.auto_wire_protocols.port_definitions.create_port_for_bufferless_design
```
````

````{py:function} create_port_for_buffered_design(*, x_width: int, y_width: int, x_count: int, y_count: int, x_address_width: typing.Optional[int] = None, y_address_width: typing.Optional[int] = None) -> elasticai.creator.vhdl.design.ports.Port
:canonical: elasticai.creator.vhdl.auto_wire_protocols.port_definitions.create_port_for_buffered_design

```{autodoc2-docstring} elasticai.creator.vhdl.auto_wire_protocols.port_definitions.create_port_for_buffered_design
```
````

````{py:function} port_definition_template_for_buffered_design() -> list[str]
:canonical: elasticai.creator.vhdl.auto_wire_protocols.port_definitions.port_definition_template_for_buffered_design

```{autodoc2-docstring} elasticai.creator.vhdl.auto_wire_protocols.port_definitions.port_definition_template_for_buffered_design
```
````

````{py:function} port_definition_template_for_bufferless_design() -> list[str]
:canonical: elasticai.creator.vhdl.auto_wire_protocols.port_definitions.port_definition_template_for_bufferless_design

```{autodoc2-docstring} elasticai.creator.vhdl.auto_wire_protocols.port_definitions.port_definition_template_for_bufferless_design
```
````

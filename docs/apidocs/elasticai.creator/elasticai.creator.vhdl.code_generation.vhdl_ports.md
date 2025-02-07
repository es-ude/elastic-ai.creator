# {py:mod}`elasticai.creator.vhdl.code_generation.vhdl_ports`

```{py:module} elasticai.creator.vhdl.code_generation.vhdl_ports
```

```{autodoc2-docstring} elasticai.creator.vhdl.code_generation.vhdl_ports
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`signal_string <elasticai.creator.vhdl.code_generation.vhdl_ports.signal_string>`
  - ```{autodoc2-docstring} elasticai.creator.vhdl.code_generation.vhdl_ports.signal_string
    :summary:
    ```
* - {py:obj}`vhdl_port_definition <elasticai.creator.vhdl.code_generation.vhdl_ports.vhdl_port_definition>`
  - ```{autodoc2-docstring} elasticai.creator.vhdl.code_generation.vhdl_ports.vhdl_port_definition
    :summary:
    ```
* - {py:obj}`template_string_for_port_definition <elasticai.creator.vhdl.code_generation.vhdl_ports.template_string_for_port_definition>`
  - ```{autodoc2-docstring} elasticai.creator.vhdl.code_generation.vhdl_ports.template_string_for_port_definition
    :summary:
    ```
* - {py:obj}`wrap_lines_into_port_statement <elasticai.creator.vhdl.code_generation.vhdl_ports.wrap_lines_into_port_statement>`
  - ```{autodoc2-docstring} elasticai.creator.vhdl.code_generation.vhdl_ports.wrap_lines_into_port_statement
    :summary:
    ```
* - {py:obj}`expand_to_parameters_for_port_template <elasticai.creator.vhdl.code_generation.vhdl_ports.expand_to_parameters_for_port_template>`
  - ```{autodoc2-docstring} elasticai.creator.vhdl.code_generation.vhdl_ports.expand_to_parameters_for_port_template
    :summary:
    ```
````

### API

````{py:function} signal_string(name: str, direction: str, width: int | str) -> str
:canonical: elasticai.creator.vhdl.code_generation.vhdl_ports.signal_string

```{autodoc2-docstring} elasticai.creator.vhdl.code_generation.vhdl_ports.signal_string
```
````

````{py:function} vhdl_port_definition(p: elasticai.creator.vhdl.design.ports.Port) -> list[str]
:canonical: elasticai.creator.vhdl.code_generation.vhdl_ports.vhdl_port_definition

```{autodoc2-docstring} elasticai.creator.vhdl.code_generation.vhdl_ports.vhdl_port_definition
```
````

````{py:function} template_string_for_port_definition(p: elasticai.creator.vhdl.design.ports.Port) -> list[str]
:canonical: elasticai.creator.vhdl.code_generation.vhdl_ports.template_string_for_port_definition

```{autodoc2-docstring} elasticai.creator.vhdl.code_generation.vhdl_ports.template_string_for_port_definition
```
````

````{py:function} wrap_lines_into_port_statement(lines: collections.abc.Sequence[str]) -> list[str]
:canonical: elasticai.creator.vhdl.code_generation.vhdl_ports.wrap_lines_into_port_statement

```{autodoc2-docstring} elasticai.creator.vhdl.code_generation.vhdl_ports.wrap_lines_into_port_statement
```
````

````{py:function} expand_to_parameters_for_port_template(p: elasticai.creator.vhdl.design.ports.Port) -> dict[str, str]
:canonical: elasticai.creator.vhdl.code_generation.vhdl_ports.expand_to_parameters_for_port_template

```{autodoc2-docstring} elasticai.creator.vhdl.code_generation.vhdl_ports.expand_to_parameters_for_port_template
```
````

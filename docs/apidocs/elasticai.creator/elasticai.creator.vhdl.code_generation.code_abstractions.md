# {py:mod}`elasticai.creator.vhdl.code_generation.code_abstractions`

```{py:module} elasticai.creator.vhdl.code_generation.code_abstractions
```

```{autodoc2-docstring} elasticai.creator.vhdl.code_generation.code_abstractions
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`create_instance <elasticai.creator.vhdl.code_generation.code_abstractions.create_instance>`
  - ```{autodoc2-docstring} elasticai.creator.vhdl.code_generation.code_abstractions.create_instance
    :summary:
    ```
* - {py:obj}`create_connections_using_to_from_pairs <elasticai.creator.vhdl.code_generation.code_abstractions.create_connections_using_to_from_pairs>`
  - ```{autodoc2-docstring} elasticai.creator.vhdl.code_generation.code_abstractions.create_connections_using_to_from_pairs
    :summary:
    ```
* - {py:obj}`create_connection <elasticai.creator.vhdl.code_generation.code_abstractions.create_connection>`
  - ```{autodoc2-docstring} elasticai.creator.vhdl.code_generation.code_abstractions.create_connection
    :summary:
    ```
* - {py:obj}`create_signal_definitions <elasticai.creator.vhdl.code_generation.code_abstractions.create_signal_definitions>`
  - ```{autodoc2-docstring} elasticai.creator.vhdl.code_generation.code_abstractions.create_signal_definitions
    :summary:
    ```
* - {py:obj}`signal_definition <elasticai.creator.vhdl.code_generation.code_abstractions.signal_definition>`
  - ```{autodoc2-docstring} elasticai.creator.vhdl.code_generation.code_abstractions.signal_definition
    :summary:
    ```
* - {py:obj}`hex_representation <elasticai.creator.vhdl.code_generation.code_abstractions.hex_representation>`
  - ```{autodoc2-docstring} elasticai.creator.vhdl.code_generation.code_abstractions.hex_representation
    :summary:
    ```
* - {py:obj}`bin_representation <elasticai.creator.vhdl.code_generation.code_abstractions.bin_representation>`
  - ```{autodoc2-docstring} elasticai.creator.vhdl.code_generation.code_abstractions.bin_representation
    :summary:
    ```
* - {py:obj}`to_vhdl_binary_string <elasticai.creator.vhdl.code_generation.code_abstractions.to_vhdl_binary_string>`
  - ```{autodoc2-docstring} elasticai.creator.vhdl.code_generation.code_abstractions.to_vhdl_binary_string
    :summary:
    ```
````

### API

````{py:function} create_instance(*, name: str, entity: str, signal_mapping: dict[str, str], library: str, architecture: str = 'rtl') -> list[str]
:canonical: elasticai.creator.vhdl.code_generation.code_abstractions.create_instance

```{autodoc2-docstring} elasticai.creator.vhdl.code_generation.code_abstractions.create_instance
```
````

````{py:function} create_connections_using_to_from_pairs(mapping: dict[str, str]) -> list[str]
:canonical: elasticai.creator.vhdl.code_generation.code_abstractions.create_connections_using_to_from_pairs

```{autodoc2-docstring} elasticai.creator.vhdl.code_generation.code_abstractions.create_connections_using_to_from_pairs
```
````

````{py:function} create_connection(sink, source) -> str
:canonical: elasticai.creator.vhdl.code_generation.code_abstractions.create_connection

```{autodoc2-docstring} elasticai.creator.vhdl.code_generation.code_abstractions.create_connection
```
````

````{py:function} create_signal_definitions(prefix: str, signals: collections.abc.Sequence[elasticai.creator.vhdl.design.signal.Signal])
:canonical: elasticai.creator.vhdl.code_generation.code_abstractions.create_signal_definitions

```{autodoc2-docstring} elasticai.creator.vhdl.code_generation.code_abstractions.create_signal_definitions
```
````

````{py:function} signal_definition(*, name: str, width: int)
:canonical: elasticai.creator.vhdl.code_generation.code_abstractions.signal_definition

```{autodoc2-docstring} elasticai.creator.vhdl.code_generation.code_abstractions.signal_definition
```
````

````{py:function} hex_representation(hex_value: str) -> str
:canonical: elasticai.creator.vhdl.code_generation.code_abstractions.hex_representation

```{autodoc2-docstring} elasticai.creator.vhdl.code_generation.code_abstractions.hex_representation
```
````

````{py:function} bin_representation(bin_value: str) -> str
:canonical: elasticai.creator.vhdl.code_generation.code_abstractions.bin_representation

```{autodoc2-docstring} elasticai.creator.vhdl.code_generation.code_abstractions.bin_representation
```
````

````{py:function} to_vhdl_binary_string(number: int, number_of_bits: int) -> str
:canonical: elasticai.creator.vhdl.code_generation.code_abstractions.to_vhdl_binary_string

```{autodoc2-docstring} elasticai.creator.vhdl.code_generation.code_abstractions.to_vhdl_binary_string
```
````

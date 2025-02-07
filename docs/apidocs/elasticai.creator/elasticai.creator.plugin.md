# {py:mod}`elasticai.creator.plugin`

```{py:module} elasticai.creator.plugin
```

```{autodoc2-docstring} elasticai.creator.plugin
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PluginSpec <elasticai.creator.plugin.PluginSpec>`
  - ```{autodoc2-docstring} elasticai.creator.plugin.PluginSpec
    :summary:
    ```
* - {py:obj}`PluginLoader <elasticai.creator.plugin.PluginLoader>`
  - ```{autodoc2-docstring} elasticai.creator.plugin.PluginLoader
    :summary:
    ```
* - {py:obj}`PluginSymbol <elasticai.creator.plugin.PluginSymbol>`
  - ```{autodoc2-docstring} elasticai.creator.plugin.PluginSymbol
    :summary:
    ```
* - {py:obj}`SymbolFetcher <elasticai.creator.plugin.SymbolFetcher>`
  - ```{autodoc2-docstring} elasticai.creator.plugin.SymbolFetcher
    :summary:
    ```
* - {py:obj}`SymbolFetcherBuilder <elasticai.creator.plugin.SymbolFetcherBuilder>`
  - ```{autodoc2-docstring} elasticai.creator.plugin.SymbolFetcherBuilder
    :summary:
    ```
* - {py:obj}`PluginSymbolFn <elasticai.creator.plugin.PluginSymbolFn>`
  - ```{autodoc2-docstring} elasticai.creator.plugin.PluginSymbolFn
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`import_symbols <elasticai.creator.plugin.import_symbols>`
  - ```{autodoc2-docstring} elasticai.creator.plugin.import_symbols
    :summary:
    ```
* - {py:obj}`make_plugin_symbol <elasticai.creator.plugin.make_plugin_symbol>`
  - ```{autodoc2-docstring} elasticai.creator.plugin.make_plugin_symbol
    :summary:
    ```
* - {py:obj}`build_plugin_spec <elasticai.creator.plugin.build_plugin_spec>`
  - ```{autodoc2-docstring} elasticai.creator.plugin.build_plugin_spec
    :summary:
    ```
* - {py:obj}`read_plugin_dicts_from_package <elasticai.creator.plugin.read_plugin_dicts_from_package>`
  - ```{autodoc2-docstring} elasticai.creator.plugin.read_plugin_dicts_from_package
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PluginDict <elasticai.creator.plugin.PluginDict>`
  - ```{autodoc2-docstring} elasticai.creator.plugin.PluginDict
    :summary:
    ```
* - {py:obj}`PluginSpecT <elasticai.creator.plugin.PluginSpecT>`
  - ```{autodoc2-docstring} elasticai.creator.plugin.PluginSpecT
    :summary:
    ```
* - {py:obj}`P <elasticai.creator.plugin.P>`
  - ```{autodoc2-docstring} elasticai.creator.plugin.P
    :summary:
    ```
````

### API

`````{py:class} PluginSpec
:canonical: elasticai.creator.plugin.PluginSpec

```{autodoc2-docstring} elasticai.creator.plugin.PluginSpec
```

````{py:attribute} name
:canonical: elasticai.creator.plugin.PluginSpec.name
:type: str
:value: >
   None

```{autodoc2-docstring} elasticai.creator.plugin.PluginSpec.name
```

````

````{py:attribute} target_platform
:canonical: elasticai.creator.plugin.PluginSpec.target_platform
:type: str
:value: >
   None

```{autodoc2-docstring} elasticai.creator.plugin.PluginSpec.target_platform
```

````

````{py:attribute} target_runtime
:canonical: elasticai.creator.plugin.PluginSpec.target_runtime
:type: str
:value: >
   None

```{autodoc2-docstring} elasticai.creator.plugin.PluginSpec.target_runtime
```

````

````{py:attribute} version
:canonical: elasticai.creator.plugin.PluginSpec.version
:type: str
:value: >
   None

```{autodoc2-docstring} elasticai.creator.plugin.PluginSpec.version
```

````

````{py:attribute} api_version
:canonical: elasticai.creator.plugin.PluginSpec.api_version
:type: str
:value: >
   None

```{autodoc2-docstring} elasticai.creator.plugin.PluginSpec.api_version
```

````

````{py:attribute} package
:canonical: elasticai.creator.plugin.PluginSpec.package
:type: str
:value: >
   None

```{autodoc2-docstring} elasticai.creator.plugin.PluginSpec.package
```

````

`````

````{py:data} PluginDict
:canonical: elasticai.creator.plugin.PluginDict
:type: typing.TypeAlias
:value: >
   None

```{autodoc2-docstring} elasticai.creator.plugin.PluginDict
```

````

````{py:data} PluginSpecT
:canonical: elasticai.creator.plugin.PluginSpecT
:value: >
   'TypeVar(...)'

```{autodoc2-docstring} elasticai.creator.plugin.PluginSpecT
```

````

````{py:data} P
:canonical: elasticai.creator.plugin.P
:value: >
   'ParamSpec(...)'

```{autodoc2-docstring} elasticai.creator.plugin.P
```

````

`````{py:class} PluginLoader(fetch: elasticai.creator.plugin.SymbolFetcher, plugin_receiver: elasticai.creator.plugin._PlRecT)
:canonical: elasticai.creator.plugin.PluginLoader

Bases: {py:obj}`typing.Generic`\[{py:obj}`elasticai.creator.plugin._PlRecT`\]

```{autodoc2-docstring} elasticai.creator.plugin.PluginLoader
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.plugin.PluginLoader.__init__
```

````{py:method} load_from_package(package: str) -> None
:canonical: elasticai.creator.plugin.PluginLoader.load_from_package

```{autodoc2-docstring} elasticai.creator.plugin.PluginLoader.load_from_package
```

````

`````

`````{py:class} PluginSymbol
:canonical: elasticai.creator.plugin.PluginSymbol

Bases: {py:obj}`typing.Protocol`\[{py:obj}`elasticai.creator.plugin._PlRecT`\]

```{autodoc2-docstring} elasticai.creator.plugin.PluginSymbol
```

````{py:method} load_into(/, receiver) -> None
:canonical: elasticai.creator.plugin.PluginSymbol.load_into
:abstractmethod:

```{autodoc2-docstring} elasticai.creator.plugin.PluginSymbol.load_into
```

````

`````

`````{py:class} SymbolFetcher
:canonical: elasticai.creator.plugin.SymbolFetcher

Bases: {py:obj}`typing.Protocol`\[{py:obj}`elasticai.creator.plugin._PlRecT`\]

```{autodoc2-docstring} elasticai.creator.plugin.SymbolFetcher
```

````{py:method} __call__(data: collections.abc.Iterable[elasticai.creator.plugin.PluginDict]) -> collections.abc.Iterator[elasticai.creator.plugin.PluginSymbol[elasticai.creator.plugin._PlRecT]]
:canonical: elasticai.creator.plugin.SymbolFetcher.__call__
:abstractmethod:

```{autodoc2-docstring} elasticai.creator.plugin.SymbolFetcher.__call__
```

````

`````

`````{py:class} SymbolFetcherBuilder(spec_type: type[elasticai.creator.plugin.PluginSpecT])
:canonical: elasticai.creator.plugin.SymbolFetcherBuilder

Bases: {py:obj}`typing.Generic`\[{py:obj}`elasticai.creator.plugin.PluginSpecT`\, {py:obj}`elasticai.creator.plugin._PlRecT`\]

```{autodoc2-docstring} elasticai.creator.plugin.SymbolFetcherBuilder
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.plugin.SymbolFetcherBuilder.__init__
```

````{py:method} add_fn_over_iter(fn: collections.abc.Callable[[collections.abc.Iterable[elasticai.creator.plugin.PluginSpecT]], collections.abc.Iterator[elasticai.creator.plugin.PluginSymbol[elasticai.creator.plugin._PlRecT]]]) -> elasticai.creator.plugin._SymbolFetcherBuilderT
:canonical: elasticai.creator.plugin.SymbolFetcherBuilder.add_fn_over_iter

```{autodoc2-docstring} elasticai.creator.plugin.SymbolFetcherBuilder.add_fn_over_iter
```

````

````{py:method} add_fn(fn: collections.abc.Callable[[elasticai.creator.plugin.PluginSpecT], collections.abc.Iterator[elasticai.creator.plugin.PluginSymbol[elasticai.creator.plugin._PlRecT]]]) -> elasticai.creator.plugin._SymbolFetcherBuilderT
:canonical: elasticai.creator.plugin.SymbolFetcherBuilder.add_fn

```{autodoc2-docstring} elasticai.creator.plugin.SymbolFetcherBuilder.add_fn
```

````

````{py:method} build() -> elasticai.creator.plugin.SymbolFetcher[elasticai.creator.plugin._PlRecT]
:canonical: elasticai.creator.plugin.SymbolFetcherBuilder.build

```{autodoc2-docstring} elasticai.creator.plugin.SymbolFetcherBuilder.build
```

````

`````

`````{py:class} PluginSymbolFn
:canonical: elasticai.creator.plugin.PluginSymbolFn

Bases: {py:obj}`elasticai.creator.plugin.PluginSymbol`\[{py:obj}`elasticai.creator.plugin._PlRecT`\], {py:obj}`typing.Generic`\[{py:obj}`elasticai.creator.plugin._PlRecT`\, {py:obj}`elasticai.creator.plugin.P`\, {py:obj}`elasticai.creator.plugin._ReturnT`\], {py:obj}`typing.Protocol`

```{autodoc2-docstring} elasticai.creator.plugin.PluginSymbolFn
```

````{py:method} __call__(*args: elasticai.creator.plugin.P, **kwargs: elasticai.creator.plugin.P) -> elasticai.creator.plugin._ReturnT
:canonical: elasticai.creator.plugin.PluginSymbolFn.__call__
:abstractmethod:

```{autodoc2-docstring} elasticai.creator.plugin.PluginSymbolFn.__call__
```

````

`````

````{py:function} import_symbols(module: str, names: collections.abc.Iterable[str]) -> collections.abc.Iterator[elasticai.creator.plugin.PluginSymbol]
:canonical: elasticai.creator.plugin.import_symbols

```{autodoc2-docstring} elasticai.creator.plugin.import_symbols
```
````

````{py:function} make_plugin_symbol(load_into: collections.abc.Callable[[elasticai.creator.plugin._PlRecT], None], fn: collections.abc.Callable[elasticai.creator.plugin.P, elasticai.creator.plugin._T]) -> elasticai.creator.plugin.PluginSymbolFn[elasticai.creator.plugin._PlRecT, elasticai.creator.plugin.P, elasticai.creator.plugin._T]
:canonical: elasticai.creator.plugin.make_plugin_symbol

```{autodoc2-docstring} elasticai.creator.plugin.make_plugin_symbol
```
````

````{py:function} build_plugin_spec(d: elasticai.creator.plugin.PluginDict, spec_type: type[elasticai.creator.plugin.PluginSpecT]) -> elasticai.creator.plugin.PluginSpecT
:canonical: elasticai.creator.plugin.build_plugin_spec

```{autodoc2-docstring} elasticai.creator.plugin.build_plugin_spec
```
````

````{py:function} read_plugin_dicts_from_package(package: str) -> collections.abc.Iterable[elasticai.creator.plugin.PluginDict]
:canonical: elasticai.creator.plugin.read_plugin_dicts_from_package

```{autodoc2-docstring} elasticai.creator.plugin.read_plugin_dicts_from_package
```
````

```{py:exception} MissingFieldError(field_names: set[str], plugin_type: type[elasticai.creator.plugin.PluginSpecT])
:canonical: elasticai.creator.plugin.MissingFieldError

Bases: {py:obj}`Exception`

```

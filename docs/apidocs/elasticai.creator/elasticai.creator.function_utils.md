# {py:mod}`elasticai.creator.function_utils`

```{py:module} elasticai.creator.function_utils
```

```{autodoc2-docstring} elasticai.creator.function_utils
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FunctionDecorator <elasticai.creator.function_utils.FunctionDecorator>`
  - ```{autodoc2-docstring} elasticai.creator.function_utils.FunctionDecorator
    :summary:
    ```
* - {py:obj}`RegisterDescriptor <elasticai.creator.function_utils.RegisterDescriptor>`
  - ```{autodoc2-docstring} elasticai.creator.function_utils.RegisterDescriptor
    :summary:
    ```
* - {py:obj}`KeyedFunctionDispatcher <elasticai.creator.function_utils.KeyedFunctionDispatcher>`
  - ```{autodoc2-docstring} elasticai.creator.function_utils.KeyedFunctionDispatcher
    :summary:
    ```
* - {py:obj}`FunctionRegistry <elasticai.creator.function_utils.FunctionRegistry>`
  - ```{autodoc2-docstring} elasticai.creator.function_utils.FunctionRegistry
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Tin <elasticai.creator.function_utils.Tin>`
  - ```{autodoc2-docstring} elasticai.creator.function_utils.Tin
    :summary:
    ```
* - {py:obj}`Tout <elasticai.creator.function_utils.Tout>`
  - ```{autodoc2-docstring} elasticai.creator.function_utils.Tout
    :summary:
    ```
* - {py:obj}`FN <elasticai.creator.function_utils.FN>`
  - ```{autodoc2-docstring} elasticai.creator.function_utils.FN
    :summary:
    ```
````

### API

````{py:data} Tin
:canonical: elasticai.creator.function_utils.Tin
:value: >
   'TypeVar(...)'

```{autodoc2-docstring} elasticai.creator.function_utils.Tin
```

````

````{py:data} Tout
:canonical: elasticai.creator.function_utils.Tout
:value: >
   'TypeVar(...)'

```{autodoc2-docstring} elasticai.creator.function_utils.Tout
```

````

````{py:data} FN
:canonical: elasticai.creator.function_utils.FN
:value: >
   'TypeVar(...)'

```{autodoc2-docstring} elasticai.creator.function_utils.FN
```

````

`````{py:class} FunctionDecorator(callback: typing.Callable[[str, elasticai.creator.function_utils.FN], elasticai.creator.function_utils.Tout])
:canonical: elasticai.creator.function_utils.FunctionDecorator

Bases: {py:obj}`typing.Generic`\[{py:obj}`elasticai.creator.function_utils.FN`\, {py:obj}`elasticai.creator.function_utils.Tout`\]

```{autodoc2-docstring} elasticai.creator.function_utils.FunctionDecorator
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.function_utils.FunctionDecorator.__init__
```

````{py:method} __call__(arg: elasticai.creator.function_utils.FN | str, arg2: elasticai.creator.function_utils.FN | None = None, /) -> elasticai.creator.function_utils.Tout | typing.Callable[[elasticai.creator.function_utils.FN], elasticai.creator.function_utils.Tout]
:canonical: elasticai.creator.function_utils.FunctionDecorator.__call__

```{autodoc2-docstring} elasticai.creator.function_utils.FunctionDecorator.__call__
```

````

`````

`````{py:class} RegisterDescriptor
:canonical: elasticai.creator.function_utils.RegisterDescriptor

Bases: {py:obj}`typing.Generic`\[{py:obj}`elasticai.creator.function_utils.Tin`\, {py:obj}`elasticai.creator.function_utils.Tout`\]

```{autodoc2-docstring} elasticai.creator.function_utils.RegisterDescriptor
```

````{py:method} __set_name__(instance, name)
:canonical: elasticai.creator.function_utils.RegisterDescriptor.__set_name__

```{autodoc2-docstring} elasticai.creator.function_utils.RegisterDescriptor.__set_name__
```

````

````{py:method} __get__(instance, owner=None) -> elasticai.creator.function_utils.FunctionDecorator[typing.Callable[[elasticai.creator.function_utils.Tin], elasticai.creator.function_utils.Tout], typing.Callable[[elasticai.creator.function_utils.Tin], elasticai.creator.function_utils.Tout]]
:canonical: elasticai.creator.function_utils.RegisterDescriptor.__get__

```{autodoc2-docstring} elasticai.creator.function_utils.RegisterDescriptor.__get__
```

````

`````

`````{py:class} KeyedFunctionDispatcher(dispatch_key_fn: typing.Callable[[elasticai.creator.function_utils.Tin], str])
:canonical: elasticai.creator.function_utils.KeyedFunctionDispatcher

Bases: {py:obj}`typing.Generic`\[{py:obj}`elasticai.creator.function_utils.Tin`\, {py:obj}`elasticai.creator.function_utils.Tout`\]

```{autodoc2-docstring} elasticai.creator.function_utils.KeyedFunctionDispatcher
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.function_utils.KeyedFunctionDispatcher.__init__
```

````{py:attribute} register
:canonical: elasticai.creator.function_utils.KeyedFunctionDispatcher.register
:type: elasticai.creator.function_utils.RegisterDescriptor[elasticai.creator.function_utils.Tin, elasticai.creator.function_utils.Tout]
:value: >
   'RegisterDescriptor(...)'

```{autodoc2-docstring} elasticai.creator.function_utils.KeyedFunctionDispatcher.register
```

````

````{py:method} __contains__(item: str | typing.Callable[[elasticai.creator.function_utils.Tin], elasticai.creator.function_utils.Tout]) -> bool
:canonical: elasticai.creator.function_utils.KeyedFunctionDispatcher.__contains__

```{autodoc2-docstring} elasticai.creator.function_utils.KeyedFunctionDispatcher.__contains__
```

````

````{py:method} can_dispatch(item: elasticai.creator.function_utils.Tin) -> bool
:canonical: elasticai.creator.function_utils.KeyedFunctionDispatcher.can_dispatch

```{autodoc2-docstring} elasticai.creator.function_utils.KeyedFunctionDispatcher.can_dispatch
```

````

````{py:method} call(arg: elasticai.creator.function_utils.Tin) -> elasticai.creator.function_utils.Tout
:canonical: elasticai.creator.function_utils.KeyedFunctionDispatcher.call

```{autodoc2-docstring} elasticai.creator.function_utils.KeyedFunctionDispatcher.call
```

````

````{py:method} __call__(arg: elasticai.creator.function_utils.Tin) -> elasticai.creator.function_utils.Tout
:canonical: elasticai.creator.function_utils.KeyedFunctionDispatcher.__call__

```{autodoc2-docstring} elasticai.creator.function_utils.KeyedFunctionDispatcher.__call__
```

````

`````

````{py:class} FunctionRegistry(dispatch_key_fn: typing.Callable[[elasticai.creator.function_utils.Tin], str])
:canonical: elasticai.creator.function_utils.FunctionRegistry

Bases: {py:obj}`elasticai.creator.function_utils.KeyedFunctionDispatcher`

```{autodoc2-docstring} elasticai.creator.function_utils.FunctionRegistry
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.function_utils.FunctionRegistry.__init__
```

````

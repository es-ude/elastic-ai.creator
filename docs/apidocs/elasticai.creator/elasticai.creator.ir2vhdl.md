# {py:mod}`elasticai.creator.ir2vhdl`

```{py:module} elasticai.creator.ir2vhdl
```

```{autodoc2-docstring} elasticai.creator.ir2vhdl
:allowtitles:
```

## Package Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Ir2Vhdl <elasticai.creator.ir2vhdl.ir2vhdl.Ir2Vhdl>`
  -
* - {py:obj}`Edge <elasticai.creator.ir2vhdl.ir2vhdl.Edge>`
  -
* - {py:obj}`Signal <elasticai.creator.ir2vhdl.ir2vhdl.Signal>`
  -
* - {py:obj}`Instance <elasticai.creator.ir2vhdl.ir2vhdl.Instance>`
  - ```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.Instance
    :summary:
    ```
* - {py:obj}`Implementation <elasticai.creator.ir2vhdl.ir2vhdl.Implementation>`
  -
* - {py:obj}`InstanceFactory <elasticai.creator.ir2vhdl.ir2vhdl.InstanceFactory>`
  - ```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.InstanceFactory
    :summary:
    ```
* - {py:obj}`LogicSignal <elasticai.creator.ir2vhdl.ir2vhdl.LogicSignal>`
  -
* - {py:obj}`LogicVectorSignal <elasticai.creator.ir2vhdl.ir2vhdl.LogicVectorSignal>`
  -
* - {py:obj}`NullDefinedLogicSignal <elasticai.creator.ir2vhdl.ir2vhdl.NullDefinedLogicSignal>`
  -
* - {py:obj}`PluginLoader <elasticai.creator.ir2vhdl.ir2vhdl.PluginLoader>`
  - ```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.PluginLoader
    :summary:
    ```
* - {py:obj}`PluginSpec <elasticai.creator.ir2vhdl.ir2vhdl.PluginSpec>`
  -
* - {py:obj}`PortMap <elasticai.creator.ir2vhdl.ir2vhdl.PortMap>`
  - ```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.PortMap
    :summary:
    ```
* - {py:obj}`Shape <elasticai.creator.ir2vhdl.ir2vhdl.Shape>`
  - ```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.Shape
    :summary:
    ```
* - {py:obj}`VhdlNode <elasticai.creator.ir2vhdl.ir2vhdl.VhdlNode>`
  - ```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.VhdlNode
    :summary:
    ```
* - {py:obj}`EntityTemplateParameter <elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateParameter>`
  - ```{autodoc2-docstring} elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateParameter
    :summary:
    ```
* - {py:obj}`ValueTemplateParameter <elasticai.creator.ir2vhdl.vhdl_template.ValueTemplateParameter>`
  - ```{autodoc2-docstring} elasticai.creator.ir2vhdl.vhdl_template.ValueTemplateParameter
    :summary:
    ```
* - {py:obj}`EntityTemplateDirector <elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateDirector>`
  - ```{autodoc2-docstring} elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateDirector
    :summary:
    ```
* - {py:obj}`Template <elasticai.creator.template.Template>`
  -
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`edge <elasticai.creator.ir2vhdl.ir2vhdl.edge>`
  - ```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.edge
    :summary:
    ```
* - {py:obj}`vhdl_node <elasticai.creator.ir2vhdl.ir2vhdl.vhdl_node>`
  - ```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.vhdl_node
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Code <elasticai.creator.ir2vhdl.ir2vhdl.Code>`
  - ```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.Code
    :summary:
    ```
* - {py:obj}`PluginSymbol <elasticai.creator.ir2vhdl.ir2vhdl.PluginSymbol>`
  - ```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.PluginSymbol
    :summary:
    ```
* - {py:obj}`ShapeTuple <elasticai.creator.ir2vhdl.ir2vhdl.ShapeTuple>`
  - ```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.ShapeTuple
    :summary:
    ```
* - {py:obj}`TypeHandlerFn <elasticai.creator.ir2vhdl.ir2vhdl.TypeHandlerFn>`
  - ```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.TypeHandlerFn
    :summary:
    ```
* - {py:obj}`type_handler <elasticai.creator.ir2vhdl.ir2vhdl.type_handler>`
  - ```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.type_handler
    :summary:
    ```
* - {py:obj}`type_handler_iterable <elasticai.creator.ir2vhdl.ir2vhdl.type_handler_iterable>`
  - ```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.type_handler_iterable
    :summary:
    ```
````

### API

`````{py:class} Ir2Vhdl()
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.Ir2Vhdl

Bases: {py:obj}`elasticai.creator.ir.LoweringPass`\[{py:obj}`elasticai.creator.ir2vhdl.ir2vhdl.Implementation`\, {py:obj}`elasticai.creator.ir2vhdl.ir2vhdl.Code`\]

````{py:method} register_static(name: str, fn: collections.abc.Callable[[], str]) -> None
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.Ir2Vhdl.register_static

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.Ir2Vhdl.register_static
```

````

````{py:method} __call__(args: collections.abc.Iterable[elasticai.creator.ir2vhdl.ir2vhdl.Implementation]) -> typing.Iterator[elasticai.creator.ir2vhdl.ir2vhdl.Code]
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.Ir2Vhdl.__call__

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.Ir2Vhdl.__call__
```

````

`````

`````{py:class} Edge(data: dict[str, elasticai.creator.ir.attribute.Attribute])
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.Edge

Bases: {py:obj}`elasticai.creator.ir.Edge`

````{py:attribute} src_sink_indices
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.Edge.src_sink_indices
:type: tuple[tuple[int, int], ...]
:value: >
   None

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.Edge.src_sink_indices
```

````

````{py:method} __hash__() -> int
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.Edge.__hash__

````

`````

````{py:data} Code
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.Code
:type: typing.TypeAlias
:value: >
   None

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.Code
```

````

`````{py:class} Signal
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.Signal

Bases: {py:obj}`abc.ABC`

````{py:attribute} types
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.Signal.types
:type: set[type[elasticai.creator.ir2vhdl.ir2vhdl.Signal]]
:value: >
   'set(...)'

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.Signal.types
```

````

````{py:method} define() -> typing.Iterator[str]
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.Signal.define
:abstractmethod:

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.Signal.define
```

````

````{py:property} name
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.Signal.name
:abstractmethod:
:type: str

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.Signal.name
```

````

````{py:method} from_code(code: str) -> elasticai.creator.ir2vhdl.ir2vhdl.Signal
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.Signal.from_code
:classmethod:

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.Signal.from_code
```

````

````{py:method} can_create_from_code(code: str) -> bool
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.Signal.can_create_from_code
:abstractmethod:
:classmethod:

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.Signal.can_create_from_code
```

````

````{py:method} register_type(t: type[elasticai.creator.ir2vhdl.ir2vhdl.Signal]) -> None
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.Signal.register_type
:classmethod:

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.Signal.register_type
```

````

````{py:method} make_instance_specific(instance: str) -> elasticai.creator.ir2vhdl.ir2vhdl.Signal
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.Signal.make_instance_specific
:abstractmethod:

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.Signal.make_instance_specific
```

````

`````

`````{py:class} Instance(node: elasticai.creator.ir2vhdl.ir2vhdl.VhdlNode, generic_map: dict[str, str], port_map: dict[str, elasticai.creator.ir2vhdl.ir2vhdl.Signal])
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.Instance

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.Instance
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.Instance.__init__
```

````{py:property} input_shape
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.Instance.input_shape
:type: elasticai.creator.ir2vhdl.ir2vhdl.Shape

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.Instance.input_shape
```

````

````{py:property} output_shape
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.Instance.output_shape
:type: elasticai.creator.ir2vhdl.ir2vhdl.Shape

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.Instance.output_shape
```

````

````{py:property} name
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.Instance.name
:type: str

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.Instance.name
```

````

````{py:property} implementation
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.Instance.implementation
:type: str

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.Instance.implementation
```

````

````{py:method} define_signals() -> typing.Iterator[str]
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.Instance.define_signals

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.Instance.define_signals
```

````

````{py:method} instantiate() -> typing.Iterator[str]
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.Instance.instantiate

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.Instance.instantiate
```

````

`````

`````{py:class} Implementation(name: str, type: str, attributes: dict[str, typing.Any], nodes=tuple(), edges=tuple())
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.Implementation

Bases: {py:obj}`elasticai.creator.ir.Graph`\[{py:obj}`elasticai.creator.ir2vhdl.ir2vhdl.N`\, {py:obj}`elasticai.creator.ir2vhdl.ir2vhdl.E`\], {py:obj}`elasticai.creator.ir.Lowerable`

````{py:property} type
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.Implementation.type
:type: str

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.Implementation.type
```

````

````{py:property} name
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.Implementation.name
:type: str

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.Implementation.name
```

````

````{py:method} asdict() -> dict[str, typing.Any]
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.Implementation.asdict

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.Implementation.asdict
```

````

````{py:method} fromdict(data: dict[str, typing.Any]) -> elasticai.creator.ir2vhdl.ir2vhdl.Implementation
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.Implementation.fromdict
:classmethod:

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.Implementation.fromdict
```

````

````{py:method} iterate_bfs_up_from(node: str) -> typing.Iterator[elasticai.creator.ir2vhdl.ir2vhdl.N]
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.Implementation.iterate_bfs_up_from

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.Implementation.iterate_bfs_up_from
```

````

`````

````{py:class} InstanceFactory()
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.InstanceFactory

Bases: {py:obj}`elasticai.creator.function_utils.KeyedFunctionDispatcher`\[{py:obj}`elasticai.creator.ir2vhdl.ir2vhdl.VhdlNode`\, {py:obj}`elasticai.creator.ir2vhdl.ir2vhdl.Instance`\]

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.InstanceFactory
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.InstanceFactory.__init__
```

````

`````{py:class} LogicSignal(name: str)
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.LogicSignal

Bases: {py:obj}`elasticai.creator.ir2vhdl.ir2vhdl.Signal`

````{py:method} define() -> typing.Iterator[str]
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.LogicSignal.define

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.LogicSignal.define
```

````

````{py:property} name
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.LogicSignal.name
:type: str

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.LogicSignal.name
```

````

````{py:method} can_create_from_code(code: str) -> bool
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.LogicSignal.can_create_from_code
:classmethod:

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.LogicSignal.can_create_from_code
```

````

````{py:method} from_code(code: str) -> elasticai.creator.ir2vhdl.ir2vhdl.Signal
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.LogicSignal.from_code
:classmethod:

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.LogicSignal.from_code
```

````

````{py:method} make_instance_specific(instance: str) -> elasticai.creator.ir2vhdl.ir2vhdl.Signal
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.LogicSignal.make_instance_specific

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.LogicSignal.make_instance_specific
```

````

````{py:method} __eq__(other: object) -> bool
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.LogicSignal.__eq__

````

`````

`````{py:class} LogicVectorSignal(name: str, width: int)
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.LogicVectorSignal

Bases: {py:obj}`elasticai.creator.ir2vhdl.ir2vhdl.Signal`

````{py:method} define() -> typing.Iterator[str]
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.LogicVectorSignal.define

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.LogicVectorSignal.define
```

````

````{py:property} name
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.LogicVectorSignal.name
:type: str

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.LogicVectorSignal.name
```

````

````{py:property} width
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.LogicVectorSignal.width
:type: int

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.LogicVectorSignal.width
```

````

````{py:method} can_create_from_code(code: str) -> bool
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.LogicVectorSignal.can_create_from_code
:classmethod:

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.LogicVectorSignal.can_create_from_code
```

````

````{py:method} from_code(code: str) -> elasticai.creator.ir2vhdl.ir2vhdl.Signal
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.LogicVectorSignal.from_code
:classmethod:

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.LogicVectorSignal.from_code
```

````

````{py:method} make_instance_specific(instance: str) -> elasticai.creator.ir2vhdl.ir2vhdl.Signal
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.LogicVectorSignal.make_instance_specific

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.LogicVectorSignal.make_instance_specific
```

````

````{py:method} __eq__(other) -> bool
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.LogicVectorSignal.__eq__

````

`````

`````{py:class} NullDefinedLogicSignal(name)
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.NullDefinedLogicSignal

Bases: {py:obj}`elasticai.creator.ir2vhdl.ir2vhdl.Signal`

````{py:method} define() -> typing.Iterator[str]
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.NullDefinedLogicSignal.define

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.NullDefinedLogicSignal.define
```

````

````{py:property} name
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.NullDefinedLogicSignal.name
:type: str

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.NullDefinedLogicSignal.name
```

````

````{py:method} can_create_from_code(code: str) -> bool
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.NullDefinedLogicSignal.can_create_from_code
:classmethod:

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.NullDefinedLogicSignal.can_create_from_code
```

````

````{py:method} from_code(code: str) -> elasticai.creator.ir2vhdl.ir2vhdl.Signal
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.NullDefinedLogicSignal.from_code
:classmethod:

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.NullDefinedLogicSignal.from_code
```

````

````{py:method} make_instance_specific(instance: str) -> elasticai.creator.ir2vhdl.ir2vhdl.Signal
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.NullDefinedLogicSignal.make_instance_specific

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.NullDefinedLogicSignal.make_instance_specific
```

````

`````

`````{py:class} PluginLoader(lowering: elasticai.creator.ir2vhdl.ir2vhdl.Ir2Vhdl)
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.PluginLoader

Bases: {py:obj}`elasticai.creator.plugin.PluginLoader`\[{py:obj}`elasticai.creator.ir2vhdl.ir2vhdl.Ir2Vhdl`\]

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.PluginLoader
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.PluginLoader.__init__
```

````{py:method} load_from_package(package: str) -> None
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.PluginLoader.load_from_package

````

`````

`````{py:class} PluginSpec
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.PluginSpec

Bases: {py:obj}`elasticai.creator.plugin.PluginSpec`

````{py:attribute} generated
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.PluginSpec.generated
:type: tuple[str, ...]
:value: >
   None

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.PluginSpec.generated
```

````

````{py:attribute} static_files
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.PluginSpec.static_files
:type: tuple[str, ...]
:value: >
   None

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.PluginSpec.static_files
```

````

`````

````{py:data} PluginSymbol
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.PluginSymbol
:type: typing.TypeAlias
:value: >
   None

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.PluginSymbol
```

````

`````{py:class} PortMap(map: dict[str, elasticai.creator.ir2vhdl.ir2vhdl.Signal])
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.PortMap

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.PortMap
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.PortMap.__init__
```

````{py:method} as_dict() -> dict[str, str]
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.PortMap.as_dict

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.PortMap.as_dict
```

````

````{py:method} from_dict(data: dict[str, str]) -> elasticai.creator.ir2vhdl.ir2vhdl.PortMap
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.PortMap.from_dict
:classmethod:

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.PortMap.from_dict
```

````

````{py:method} __eq__(other: object) -> bool
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.PortMap.__eq__

````

`````

`````{py:class} Shape(*values: int)
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.Shape

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.Shape
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.Shape.__init__
```

````{py:method} from_tuple(values: elasticai.creator.ir2vhdl.ir2vhdl.ShapeTuple) -> elasticai.creator.ir2vhdl.ir2vhdl.Shape
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.Shape.from_tuple
:classmethod:

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.Shape.from_tuple
```

````

````{py:method} to_tuple() -> elasticai.creator.ir2vhdl.ir2vhdl.ShapeTuple
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.Shape.to_tuple

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.Shape.to_tuple
```

````

````{py:method} __getitem__(item)
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.Shape.__getitem__

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.Shape.__getitem__
```

````

````{py:method} size() -> int
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.Shape.size

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.Shape.size
```

````

````{py:method} ndim() -> int
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.Shape.ndim

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.Shape.ndim
```

````

````{py:property} depth
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.Shape.depth
:type: int

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.Shape.depth
```

````

````{py:method} __eq__(other)
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.Shape.__eq__

````

````{py:property} width
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.Shape.width
:type: int

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.Shape.width
```

````

````{py:property} height
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.Shape.height
:type: int

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.Shape.height
```

````

````{py:method} __repr__() -> str
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.Shape.__repr__

````

`````

````{py:data} ShapeTuple
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.ShapeTuple
:type: typing.TypeAlias
:value: >
   None

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.ShapeTuple
```

````

````{py:data} TypeHandlerFn
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.TypeHandlerFn
:type: typing.TypeAlias
:value: >
   None

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.TypeHandlerFn
```

````

`````{py:class} VhdlNode(data: dict[str, elasticai.creator.ir.attribute.Attribute])
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.VhdlNode

Bases: {py:obj}`elasticai.creator.ir.Node`

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.VhdlNode
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.VhdlNode.__init__
```

````{py:attribute} implementation
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.VhdlNode.implementation
:type: str
:value: >
   None

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.VhdlNode.implementation
```

````

````{py:attribute} input_shape
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.VhdlNode.input_shape
:type: elasticai.creator.ir.RequiredField[elasticai.creator.ir2vhdl.ir2vhdl.ShapeTuple, elasticai.creator.ir2vhdl.ir2vhdl.Shape]
:value: >
   'ShapeField(...)'

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.VhdlNode.input_shape
```

````

````{py:attribute} output_shape
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.VhdlNode.output_shape
:type: elasticai.creator.ir.RequiredField[elasticai.creator.ir2vhdl.ir2vhdl.ShapeTuple, elasticai.creator.ir2vhdl.ir2vhdl.Shape]
:value: >
   'ShapeField(...)'

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.VhdlNode.output_shape
```

````

`````

````{py:function} edge(src: str, sink: str, src_sink_indices: collections.abc.Iterable[tuple[int, int]] | tuple[str, str]) -> elasticai.creator.ir2vhdl.ir2vhdl.Edge
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.edge

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.edge
```
````

````{py:data} type_handler
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.type_handler
:value: >
   'FunctionDecorator(...)'

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.type_handler
```

````

````{py:data} type_handler_iterable
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.type_handler_iterable
:value: >
   'FunctionDecorator(...)'

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.type_handler_iterable
```

````

````{py:function} vhdl_node(name: str, type: str, implementation: str, input_shape: elasticai.creator.ir2vhdl.ir2vhdl.Shape | elasticai.creator.ir2vhdl.ir2vhdl.ShapeTuple, output_shape: elasticai.creator.ir2vhdl.ir2vhdl.Shape | elasticai.creator.ir2vhdl.ir2vhdl.ShapeTuple, attributes: dict | None = None) -> elasticai.creator.ir2vhdl.ir2vhdl.VhdlNode
:canonical: elasticai.creator.ir2vhdl.ir2vhdl.vhdl_node

```{autodoc2-docstring} elasticai.creator.ir2vhdl.ir2vhdl.vhdl_node
```
````

`````{py:class} EntityTemplateParameter()
:canonical: elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateParameter

Bases: {py:obj}`elasticai.creator.template.AnalysingTemplateParameterType`

```{autodoc2-docstring} elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateParameter
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateParameter.__init__
```

````{py:method} analyse(m: re.Match) -> None
:canonical: elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateParameter.analyse

```{autodoc2-docstring} elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateParameter.analyse
```

````

````{py:method} replace(m: re.Match)
:canonical: elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateParameter.replace

```{autodoc2-docstring} elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateParameter.replace
```

````

`````

`````{py:class} ValueTemplateParameter()
:canonical: elasticai.creator.ir2vhdl.vhdl_template.ValueTemplateParameter

Bases: {py:obj}`elasticai.creator.template.TemplateParameterType`

```{autodoc2-docstring} elasticai.creator.ir2vhdl.vhdl_template.ValueTemplateParameter
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.ir2vhdl.vhdl_template.ValueTemplateParameter.__init__
```

````{py:method} replace(m: re.Match) -> str
:canonical: elasticai.creator.ir2vhdl.vhdl_template.ValueTemplateParameter.replace

```{autodoc2-docstring} elasticai.creator.ir2vhdl.vhdl_template.ValueTemplateParameter.replace
```

````

`````

`````{py:class} EntityTemplateDirector()
:canonical: elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateDirector

```{autodoc2-docstring} elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateDirector
```

```{rubric} Initialization
```

```{autodoc2-docstring} elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateDirector.__init__
```

````{py:method} set_prototype(prototype: str) -> elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateDirector
:canonical: elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateDirector.set_prototype

```{autodoc2-docstring} elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateDirector.set_prototype
```

````

````{py:method} add_generic(name: str) -> elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateDirector
:canonical: elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateDirector.add_generic

```{autodoc2-docstring} elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateDirector.add_generic
```

````

````{py:method} build() -> elasticai.creator.template.Template
:canonical: elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateDirector.build

```{autodoc2-docstring} elasticai.creator.ir2vhdl.vhdl_template.EntityTemplateDirector.build
```

````

`````

`````{py:class} Template
:canonical: elasticai.creator.template.Template

Bases: {py:obj}`typing.Protocol`

````{py:method} render(mapping: dict[str, elasticai.creator.template.TemplateParameter]) -> str
:canonical: elasticai.creator.template.Template.render

```{autodoc2-docstring} elasticai.creator.template.Template.render
```

````

`````

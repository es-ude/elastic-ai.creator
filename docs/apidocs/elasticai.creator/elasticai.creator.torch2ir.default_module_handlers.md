# {py:mod}`elasticai.creator.torch2ir.default_module_handlers`

```{py:module} elasticai.creator.torch2ir.default_module_handlers
```

```{autodoc2-docstring} elasticai.creator.torch2ir.default_module_handlers
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`conv1d <elasticai.creator.torch2ir.default_module_handlers.conv1d>`
  - ```{autodoc2-docstring} elasticai.creator.torch2ir.default_module_handlers.conv1d
    :summary:
    ```
* - {py:obj}`maxpool1d <elasticai.creator.torch2ir.default_module_handlers.maxpool1d>`
  - ```{autodoc2-docstring} elasticai.creator.torch2ir.default_module_handlers.maxpool1d
    :summary:
    ```
* - {py:obj}`linear <elasticai.creator.torch2ir.default_module_handlers.linear>`
  - ```{autodoc2-docstring} elasticai.creator.torch2ir.default_module_handlers.linear
    :summary:
    ```
* - {py:obj}`batchnorm1d <elasticai.creator.torch2ir.default_module_handlers.batchnorm1d>`
  - ```{autodoc2-docstring} elasticai.creator.torch2ir.default_module_handlers.batchnorm1d
    :summary:
    ```
* - {py:obj}`flatten <elasticai.creator.torch2ir.default_module_handlers.flatten>`
  - ```{autodoc2-docstring} elasticai.creator.torch2ir.default_module_handlers.flatten
    :summary:
    ```
* - {py:obj}`relu <elasticai.creator.torch2ir.default_module_handlers.relu>`
  - ```{autodoc2-docstring} elasticai.creator.torch2ir.default_module_handlers.relu
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`handlers <elasticai.creator.torch2ir.default_module_handlers.handlers>`
  - ```{autodoc2-docstring} elasticai.creator.torch2ir.default_module_handlers.handlers
    :summary:
    ```
````

### API

````{py:data} handlers
:canonical: elasticai.creator.torch2ir.default_module_handlers.handlers
:value: >
   []

```{autodoc2-docstring} elasticai.creator.torch2ir.default_module_handlers.handlers
```

````

````{py:function} conv1d(module: torch.nn.Conv1d) -> dict
:canonical: elasticai.creator.torch2ir.default_module_handlers.conv1d

```{autodoc2-docstring} elasticai.creator.torch2ir.default_module_handlers.conv1d
```
````

````{py:function} maxpool1d(module: torch.nn.MaxPool1d) -> dict
:canonical: elasticai.creator.torch2ir.default_module_handlers.maxpool1d

```{autodoc2-docstring} elasticai.creator.torch2ir.default_module_handlers.maxpool1d
```
````

````{py:function} linear(module: torch.nn.Linear) -> dict
:canonical: elasticai.creator.torch2ir.default_module_handlers.linear

```{autodoc2-docstring} elasticai.creator.torch2ir.default_module_handlers.linear
```
````

````{py:function} batchnorm1d(module: torch.nn.BatchNorm1d) -> dict
:canonical: elasticai.creator.torch2ir.default_module_handlers.batchnorm1d

```{autodoc2-docstring} elasticai.creator.torch2ir.default_module_handlers.batchnorm1d
```
````

````{py:function} flatten(module: torch.nn.Flatten) -> dict
:canonical: elasticai.creator.torch2ir.default_module_handlers.flatten

```{autodoc2-docstring} elasticai.creator.torch2ir.default_module_handlers.flatten
```
````

````{py:function} relu(module: torch.nn.ReLU) -> dict
:canonical: elasticai.creator.torch2ir.default_module_handlers.relu

```{autodoc2-docstring} elasticai.creator.torch2ir.default_module_handlers.relu
```
````

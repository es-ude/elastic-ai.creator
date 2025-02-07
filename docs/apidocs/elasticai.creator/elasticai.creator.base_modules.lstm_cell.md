# {py:mod}`elasticai.creator.base_modules.lstm_cell`

```{py:module} elasticai.creator.base_modules.lstm_cell
```

```{autodoc2-docstring} elasticai.creator.base_modules.lstm_cell
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MathOperations <elasticai.creator.base_modules.lstm_cell.MathOperations>`
  -
* - {py:obj}`LSTMCell <elasticai.creator.base_modules.lstm_cell.LSTMCell>`
  -
````

### API

```{py:class} MathOperations
:canonical: elasticai.creator.base_modules.lstm_cell.MathOperations

Bases: {py:obj}`elasticai.creator.base_modules.math_operations.Quantize`, {py:obj}`elasticai.creator.base_modules.math_operations.Add`, {py:obj}`elasticai.creator.base_modules.math_operations.MatMul`, {py:obj}`elasticai.creator.base_modules.math_operations.Mul`, {py:obj}`typing.Protocol`

```

`````{py:class} LSTMCell(input_size: int, hidden_size: int, bias: bool, operations: elasticai.creator.base_modules.lstm_cell.MathOperations, sigmoid_factory: collections.abc.Callable[[], torch.nn.Module], tanh_factory: collections.abc.Callable[[], torch.nn.Module], device: typing.Any = None)
:canonical: elasticai.creator.base_modules.lstm_cell.LSTMCell

Bases: {py:obj}`torch.nn.Module`

````{py:method} forward(x: torch.Tensor, state: typing.Optional[tuple[torch.Tensor, torch.Tensor]] = None) -> tuple[torch.Tensor, torch.Tensor]
:canonical: elasticai.creator.base_modules.lstm_cell.LSTMCell.forward

```{autodoc2-docstring} elasticai.creator.base_modules.lstm_cell.LSTMCell.forward
```

````

`````

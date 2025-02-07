# {py:mod}`elasticai.creator.base_modules.lstm`

```{py:module} elasticai.creator.base_modules.lstm
```

```{autodoc2-docstring} elasticai.creator.base_modules.lstm
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LayerFactory <elasticai.creator.base_modules.lstm.LayerFactory>`
  -
* - {py:obj}`LSTM <elasticai.creator.base_modules.lstm.LSTM>`
  -
````

### API

`````{py:class} LayerFactory
:canonical: elasticai.creator.base_modules.lstm.LayerFactory

Bases: {py:obj}`typing.Protocol`

````{py:method} lstm(input_size: int, hidden_size: int, bias: bool) -> elasticai.creator.base_modules.lstm_cell.LSTMCell
:canonical: elasticai.creator.base_modules.lstm.LayerFactory.lstm

```{autodoc2-docstring} elasticai.creator.base_modules.lstm.LayerFactory.lstm
```

````

`````

`````{py:class} LSTM(input_size: int, hidden_size, bias: bool, batch_first: bool, layers: elasticai.creator.base_modules.lstm.LayerFactory)
:canonical: elasticai.creator.base_modules.lstm.LSTM

Bases: {py:obj}`torch.nn.Module`

````{py:property} hidden_size
:canonical: elasticai.creator.base_modules.lstm.LSTM.hidden_size
:type: int

```{autodoc2-docstring} elasticai.creator.base_modules.lstm.LSTM.hidden_size
```

````

````{py:property} input_size
:canonical: elasticai.creator.base_modules.lstm.LSTM.input_size
:type: int

```{autodoc2-docstring} elasticai.creator.base_modules.lstm.LSTM.input_size
```

````

````{py:method} forward(x: torch.Tensor, state: typing.Optional[tuple[torch.Tensor, torch.Tensor]] = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
:canonical: elasticai.creator.base_modules.lstm.LSTM.forward

```{autodoc2-docstring} elasticai.creator.base_modules.lstm.LSTM.forward
```

````

`````

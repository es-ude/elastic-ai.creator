# {py:mod}`elasticai.creator.nn.fixed_point.lstm.design.fp_lstm_cell`

```{py:module} elasticai.creator.nn.fixed_point.lstm.design.fp_lstm_cell
```

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.lstm.design.fp_lstm_cell
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FPLSTMCell <elasticai.creator.nn.fixed_point.lstm.design.fp_lstm_cell.FPLSTMCell>`
  -
````

### API

`````{py:class} FPLSTMCell(*, name: str, hardtanh: elasticai.creator.nn.fixed_point.lstm.design._common_imports.Design, hardsigmoid: elasticai.creator.nn.fixed_point.lstm.design._common_imports.Design, total_bits: int, frac_bits: int, w_ih: list[list[list[int]]], w_hh: list[list[list[int]]], b_ih: list[list[int]], b_hh: list[list[int]])
:canonical: elasticai.creator.nn.fixed_point.lstm.design.fp_lstm_cell.FPLSTMCell

Bases: {py:obj}`elasticai.creator.nn.fixed_point.lstm.design._common_imports.Design`

````{py:property} total_bits
:canonical: elasticai.creator.nn.fixed_point.lstm.design.fp_lstm_cell.FPLSTMCell.total_bits
:type: int

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.lstm.design.fp_lstm_cell.FPLSTMCell.total_bits
```

````

````{py:property} frac_bits
:canonical: elasticai.creator.nn.fixed_point.lstm.design.fp_lstm_cell.FPLSTMCell.frac_bits
:type: int

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.lstm.design.fp_lstm_cell.FPLSTMCell.frac_bits
```

````

````{py:property} port
:canonical: elasticai.creator.nn.fixed_point.lstm.design.fp_lstm_cell.FPLSTMCell.port
:type: elasticai.creator.nn.fixed_point.lstm.design._common_imports.Port

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.lstm.design.fp_lstm_cell.FPLSTMCell.port
```

````

````{py:method} get_file_load_order() -> list[str]
:canonical: elasticai.creator.nn.fixed_point.lstm.design.fp_lstm_cell.FPLSTMCell.get_file_load_order

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.lstm.design.fp_lstm_cell.FPLSTMCell.get_file_load_order
```

````

````{py:method} save_to(destination: elasticai.creator.nn.fixed_point.lstm.design._common_imports.Path) -> None
:canonical: elasticai.creator.nn.fixed_point.lstm.design.fp_lstm_cell.FPLSTMCell.save_to

```{autodoc2-docstring} elasticai.creator.nn.fixed_point.lstm.design.fp_lstm_cell.FPLSTMCell.save_to
```

````

`````

## ElasticAI.creator

--------

## Developing a new HW-SW Module

------

## Simulation Based Testing

```python
def linear_layer_test():
  layer = Linear(in_features=2, out_features=1, total_bits=4, frac_bits=0)
  layer.weights.data = torch.tensor([[3, -1]])
  input = torch.tensor([[-2, 3]])
  expected_output = layer(input)
  design_under_test = layer.create_design("linear")
  io_converter = layer.create_io_converter_for_simulation()
  test_bench = TestBench(inputs=io_converter.torch_to_vhdl(input),
                         design_under_test=design_under_test)
  runner = NetworkModuleSimulationRunner(test_bench, workdir="build")
  runner.prepare()
  runner.run()
  result = runner.getResults()
  actual = io_converter.vhdl_to_torch(result)
  assert expected_output == expected_output
```

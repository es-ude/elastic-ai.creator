"""
Test list:
- make sure that the create_input_data functions from elasticai.creator.derive_datasets
  are used correctly
- make sure precomputations are serializable and loadable
- depthwise convolutions are handled correctly

Extra:
- compare given input_domains shape and rank to the input_shape and reuse and
  use input_domain's elements to build a set of tensors of shape input_shape. E.g.:
  input_shape = (2, 2)
  input_domain = [(1, 1), (-1, 1)]
  expected_result = [((1, 1), (1, 1)),
                     ((1, 1), (-1, 1)),
                     ((-1, 1), (1, 1)),
                     ((-1, 1), (-1, 1))]
"""

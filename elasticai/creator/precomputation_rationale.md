# Design Rationale for Precomputation API

## Dataflow Specification and IOTable Formats

### Intralayer Dataflow
Our approach considers neural network layers to be black boxes, but it is essential to know for the hardware implementation how these boxes need to be connected to each other. Additionally there is a discrepancy between the precomputation in software and the way the precomputed results will have to be considered for the hardware implementation. As an example consider a depthwise convolution. In pytorch we can achieve that by specifying e.g.
```python
conv = Conv1d(input_channels=3, groups=3, output_channels=3, kernel_size=3)
```
Now let's assume that layer receives a tensor
```python
t = torch.tensor([[2, 2, 2], [1, 1, 1], [3, 3, 3]])
```
and the weights of that layer are all equal to `1`. In this case we will obtain
```python
>>> conv(t)
tensor([[6], [3], [9]])
```
But we could have obtained the same result by slicing the tensor `t` along its channel axis then processing each channel individually and stacking the results again. For the hardware generating tool it is crucial to know the smallest possible unit the layer can be decomposed into.

We could try to get that information from the pytorch modules automatically, however this poses a drastic development overhead taking human resources we do not have. Therefore we postpone this task. Instead we want to define an easy to use API allowing developers to specify information about how inputs and outputs of the layer are connected, ie. how the layer function can be decomposed into smaller functions.
To cover the case above we could just use a structure like
```
((3, 1), (3, 1), (3, 1))
```
or
```
         conv
           |
   --------+-------
   |       |       |
subconv subconv subconv
 in=3     in=3     in=3
 out=1    out=1    out=1
```

This structure can already be represented by structuring the format of the `IOTable` data structure correctly.
An `IOTable` therefore is a sequence of tuples containing inputs and outputs of a specific layer, e.g. one entry for the above convolution could look like this
```python
subconv_entry = ((0, 0, 0), (1,))
```
whereas a partially defined convolution where the output for only two inputs is specified would look like this
```python
partial_subconv_io_table = (
    ((0, 0, 0), (1,)),
    ((0, 0, 1), (0,)),
)
```
a possible way to implement this is to use tensors or numpy arrays directly and specify
```python
partial_subconv_inputs = torch.tensor([[0, 0, 0], [0, 0, 1]])
parital_subconv_outputs = torch.tensor([[1], [0]])
```
this makes especially sense, because the outputs will have to be calculated from the inputs as
```python
outputs = layer(inputs)
```
However as mentioned above there is a discrepancy between the software neural network layer and the tables we want to generate. We want to and can decompose the convolution above into three smaller tables, however the software interface provided by the ml frameworks is hiding that fact to great extent. Therefore we will decorate the ml frameworks classes to carry information that will allow us to decompose the tables accordingly.

### Interlayer Dataflow

Additionally we need to express how outputs of one table are connected to the inputs of another table. Contrary to the above *intralayer dataflow* this will also involve buffering and striding.
class DataFlowSpecification:
    """
    The class is used to model how data flows from one layer to the next, but also to express how data flows through
    a layer. E.g. the dataflow for a mapping corresponding to a
    matrix $A = \begin{pmatrix} 1 & 0 & 1 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}$
    modelled by a `DataFlowSpecification` `d` could be printed with
    ```
    for item in d:
      print(f"{item} : {d[item]}")
    ```
    and would result in the following output
    ```
    0: (0, 2)
    1: (1,)
    2: (2,)
    ```
    ```
    outputs = data_flow_spec[input]
    ```
    """

    def __init__(self):
        raise NotImplementedError

    def __getitem__(self, input: int) -> tuple[int]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __next__(self) -> int:
        raise NotImplementedError

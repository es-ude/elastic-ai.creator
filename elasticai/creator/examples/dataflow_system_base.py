from functools import partial

from torch.nn import Sequential, MaxPool1d, Conv1d

from elasticai.creator.dataflow import DataSource, DataSink,InterLayerDataflow
from elasticai.creator.input_domains import create_depthwise_input_for_1d_conv, create_input_for_1d_conv
from elasticai.creator.io_table import IOTable, group_tables
from elasticai.creator.layers import  Binarize, ChannelShuffle
from elasticai.creator.precomputation import precomputable, get_precomputations_from_direct_children
from elasticai.creator.tags_utils import tag, get_tags

model = Sequential(
    precomputable(
        Sequential(
            Conv1d(
                in_channels=2,
                out_channels=4,
                kernel_size=(4,),
                groups=2,
            ),
            Binarize(),
        ),
        input_generator=partial(
            create_depthwise_input_for_1d_conv, shape=(2, 4), items=[-1, 1]
        ),
        input_shape=(2, 2),
    ),
    MaxPool1d(kernel_size=2),
    ChannelShuffle(groups=2),
    precomputable(
        Sequential(
            Conv1d(
                in_channels=4,
                out_channels=2,
                kernel_size=(1,),
                groups=2
            ),
            Binarize(),
        ),
        input_generator=partial(create_input_for_1d_conv, shape=(1, 4), items=[-1, 1]),
        input_shape=(6, 1),
    ),

    precomputable(
        Sequential(
            Conv1d(
                in_channels=2,
                out_channels=4,
                kernel_size=(1,),
            ),
            Binarize(),
        ),
        input_generator=partial(create_input_for_1d_conv, shape=(1, 2), items=[-1, 1]),
        input_shape=(6, 1),
    ),
)
tables = []
for precompute in get_precomputations_from_direct_children(model):
    precompute()
    tables.append(IOTable(inputs=precompute.input_domain, outputs=precompute.output))

tables[0] = group_tables(table=tables[0],groups=2)
tables[1] = group_tables(table=tables[1],groups=2)
nodes = [model[0],model[3],model[4]]
for node,table in zip(nodes,tables):
    tag(node,table= table) 
sources = []
sources.append(DataSource(node = nodes[0],selection=slice(0,1)))
sources.append(DataSource(node = nodes[0],selection=slice(2,3)))
sources.append(DataSource(node = nodes[1],selection=0))
sources.append(DataSource(node = nodes[2],selection=0))
specification = InterLayerDataflow()
specification.append(DataSink(sources=[sources[0]],shape=(1,2),node=nodes[1],selection=0))
specification.append(DataSink(sources=[sources[1]],shape=(1,2),node=nodes[1],selection=1))
specification.append(DataSink(sources=[sources[2]],shape=(1,2),node=nodes[2],selection=0))
specification.append(DataSink(sources=[sources[3]],shape=(1,2),node=None,selection=0))
print(specification)
for sink in specification.sinks:
    print(f"Data sink: {sink.node}, {sink.selection}, Generating table: {get_tags(sink.sources[0].source)['table'][sink.selection]}")
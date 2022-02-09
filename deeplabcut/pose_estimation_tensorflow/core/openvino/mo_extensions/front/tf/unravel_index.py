import numpy as np

from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Graph
from mo.ops.const import Const
from mo.ops.concat import Concat
from mo.ops.strided_slice import StridedSlice
from mo.front.common.partial_infer.utils import int64_array
from extensions.ops.activation_ops import Floor
from extensions.ops.elementwise import FloorMod, Div
from extensions.ops.pack import PackOp


class UnravelIndex(FrontReplacementSubgraph):
    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('unravel_index', dict(op='UnravelIndex')),
            ],
            edges=[
            ])

    @staticmethod
    def replace_sub_graph(graph: Graph, match: dict):
        unravel_index = match['unravel_index']

        inp0 = unravel_index.in_port(0).get_source().node
        inp1 = unravel_index.in_port(1).get_source().node

        begin_id = Const(graph, {'value': int64_array([0])}).create_node()
        end_id = Const(graph, {'value': int64_array([1])}).create_node()
        dim0 = StridedSlice(graph, dict(name=inp0.name + '/dim0',
                                        begin_mask=[1],
                                        end_mask=[1],
                                        shrink_axis_mask=[0],
                                        new_axis_mask=[0],
                                        ellipsis_mask=[0])).create_node([inp1, begin_id, end_id])

        begin_id = Const(graph, {'value': int64_array([1])}).create_node()
        end_id = Const(graph, {'value': int64_array([2])}).create_node()
        dim1 = StridedSlice(graph, dict(name=inp0.name + '/dim1',
                                        begin_mask=[1],
                                        end_mask=[1],
                                        shrink_axis_mask=[0],
                                        new_axis_mask=[0],
                                        ellipsis_mask=[0])).create_node([inp1, begin_id, end_id])

        rows = Div(graph, dict(name=unravel_index.name + "/rows")).create_node([inp0, dim0])
        rows = Floor(graph, dict(name=unravel_index.name + "/rows")).create_node([rows])
        cols = FloorMod(graph, dict(name=unravel_index.name + "/cols")).create_node([inp0, dim1])


        concat = PackOp(graph, dict(name=unravel_index.name + "/merged", axis=0)).create_node([rows, cols])

        # shape = np.array([-1, 100, 29, 29, 29], dtype=np.int32)
        # shapeNode = Const(graph, {'value': shape}).create_node()
        # restore = Reshape(graph, dict(name=inp0.name + '/reshape')).create_node([new_concat, shapeNode])

        unravel_index.out_port(0).get_connection().set_source(concat.out_port(0))

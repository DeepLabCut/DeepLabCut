from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Graph, Node
from mo.ops.const import Const
from mo.ops.strided_slice import StridedSlice
from mo.front.common.partial_infer.utils import int64_array
from extensions.ops.activation_ops import Floor
from extensions.ops.elementwise import FloorMod, Div
from extensions.ops.pack import PackOp


class UnravelIndex(FrontReplacementOp):
    op = "UnravelIndex"
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        inp0 = node.in_port(0).get_source().node
        inp1 = node.in_port(1).get_source().node

        begin_id = Const(graph, {'value': int64_array([1])}).create_node()
        end_id = Const(graph, {'value': int64_array([2])}).create_node()
        dim1 = StridedSlice(graph, dict(name=inp0.name + '/dim1',
                                        begin_mask=[1],
                                        end_mask=[1],
                                        shrink_axis_mask=[0],
                                        new_axis_mask=[0],
                                        ellipsis_mask=[0])).create_node([inp1, begin_id, end_id])

        rows = Div(graph, dict(name=node.name + "/rows")).create_node([inp0, dim1])
        rows = Floor(graph, dict(name=node.name + "/rows")).create_node([rows])
        cols = FloorMod(graph, dict(name=node.name + "/cols")).create_node([inp0, dim1])

        concat = PackOp(graph, dict(name=node.name + "/merged", axis=0)).create_node([rows, cols])
        return [concat.id]

from ryven.node_env import *


class MyCustomNode(Node):
    title = "My Node"
    init_inputs = [NodeInputType()]
    init_outputs = [NodeOutputType()]

    def update_event(self, inp=-1):
        # Your custom logic here
        data = self.input(0)
        self.set_output_val(0, data * 2)


export_nodes(MyCustomNode)

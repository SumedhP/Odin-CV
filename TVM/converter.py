import onnx
from tvm import relay
from tvm.contrib import graph_executor
from tvm import runtime
import numpy as np

# Chat-GPT cooked this

# Step 1: Load the ONNX model
onnx_model_path = "model.onnx"  # Replace with your ONNX model file path
onnx_model = onnx.load(onnx_model_path)

# Verify the ONNX model
onnx.checker.check_model(onnx_model)

# Step 2: Convert ONNX model to TVM's Relay IR
input_name = "images"  # Replace with the actual input node name of your ONNX model
input_shape = (1, 3, 416, 416) 
shape_dict = {input_name: input_shape}

# Convert the ONNX model to Relay IR
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

# Step 3: Compile the model for OpenCL
target = "opencl"  # Target the OpenCL backend

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target, params=params)

# Export the compiled model (optional)
lib.export_library("model_opencl.tar")

# Step 4: Load the compiled model
device = tvm.device(target, 0)  # Use GPU device (device_id=0)
module = runtime.GraphModule(lib["default"](device))

# Step 5: Run the model
# Prepare input data
input_data = np.random.rand(*input_shape).astype("float32")
module.set_input(input_name, input_data)

# Run inference
module.run()

# Get the output
output = module.get_output(0).asnumpy()
print("Model Output:", output)

import onnx
import numpy as np

def check_precision(onnx_model_path):
    """
    Check the precision (data types) used in an ONNX model's initializers.

    Args:
        onnx_model_path (str): Path to the ONNX model.

    Returns:
        dict: A summary of data types in the model.
    """
    # Load the ONNX model
    model = onnx.load(onnx_model_path)
    onnx.checker.check_model(model)

    # Initialize counters
    precision_summary = {"float32": 0, "int8": 0, "other": 0}

    # Inspect initializers for data types
    for initializer in model.graph.initializer:
        dtype = np.dtype(onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[initializer.data_type])
        if dtype == np.float32:
            precision_summary["float32"] += 1
        elif dtype == np.int8 or dtype == np.uint8:
            precision_summary["int8"] += 1
        else:
            precision_summary["other"] += 1

    return precision_summary


def get_convolution_nodes(onnx_model_path):
    """
    Get a list of all convolution nodes in an ONNX model.

    Args:
        onnx_model_path (str): Path to the ONNX model.

    Returns:
        list: A list of names of convolution nodes.
    """
    # Load the ONNX model
    model = onnx.load(onnx_model_path)
    onnx.checker.check_model(model)

    # List to store convolution node names
    conv_nodes = []

    # Iterate through all nodes in the graph
    for node in model.graph.node:
        if node.op_type == "Conv":  # Check for convolution nodes
            conv_nodes.append(node.name)  # Add node name to the list

    return conv_nodes

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_model_to_int8(input_model_path, output_model_path):
    """
    Quantize an ONNX model to INT8 using ONNX Runtime.

    Args:
        input_model_path (str): Path to the input ONNX model.
        output_model_path (str): Path to save the quantized ONNX model.
    """
    
    convulution_nodes = get_convolution_nodes(input_model_path)
    
    # Perform dynamic quantization
    quantize_dynamic(
        model_input=input_model_path,  # Path to the input model
        model_output=output_model_path,  # Path to save the quantized model
        per_channel=False,  # Use per-channel quantization (optional, improves accuracy for some models)
        # activation_type=QuantType.QInt8,  # Quantize activations to INT8
        weight_type=QuantType.QInt8,  # Quantize weights to INT8
        nodes_to_exclude=convulution_nodes  # List of convolution node names to exclude from quantization
    )
    print(f"Quantized model saved to: {output_model_path}")


def testModel(path, bool_fp16=False):
  import onnxruntime as ort
  model = ort.InferenceSession(path)
  
  input = np.random.randn(1, 3, 416, 416).astype(np.float32 if not bool_fp16 else np.float16)

  model_input = {"images": input}

  from time import perf_counter_ns

  # Warm up
  ITERATIONS = 10
  start = perf_counter_ns()
  for i in range(ITERATIONS):
    model.run(None, model_input)
  end = perf_counter_ns()

  print(f"Inference time: {(end - start) * 1.0 / ITERATIONS / 1e6} ms")

  ITERATIONS = 100
  start = perf_counter_ns()
  for i in range(ITERATIONS):
    model.run(None, model_input)
  end = perf_counter_ns()

  print(f"Inference time: {(end - start) * 1.0 / ITERATIONS / 1e6} ms")

def makeFP16(path, output_path):
    model = onnx.load(path)
    from onnxconverter_common import float16
    model = float16.convert_float_to_float16(model)
    onnx.save(model, output_path)
     
if __name__ == "__main__":
    # Provide the path to your ONNX model
    onnx_model_path = "CV/models/model.onnx"

    convs = get_convolution_nodes(onnx_model_path)

    print("Convolution nodes found:", len(convs))

    # Check precision
    summary = check_precision(onnx_model_path)

    testModel(onnx_model_path)

    # Print results
    # print("Precision Summary:")
    # for precision, count in summary.items():
    #     print(f"{precision}: {count}")
    
    # Quantize the model to int8
    quantized = "CV/models/model_fp16.onnx"
    # quantize_model_to_int8(onnx_model_path, quantized)
    makeFP16(onnx_model_path, quantized)
    
    
    # Check precision of the quantized model
    summary = check_precision(quantized)
    
    testModel(quantized, True)
    
    # Print results
    # print("Quantized Model Precision Summary:")
    # for precision, count in summary.items():
    #     print(f"{precision}: {count}")

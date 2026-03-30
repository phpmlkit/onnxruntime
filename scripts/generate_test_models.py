#!/usr/bin/env python3
"""
Generate simple ONNX test models for the PHP ONNX Runtime test suite.

Requirements:
    pip install onnx numpy

Usage:
    python generate_test_models.py
"""

import os
import onnx
from onnx import helper, TensorProto
import numpy as np

def create_identity_model(output_dir: str) -> None:
    """Create a simple identity model: output = input"""
    
    # Define input
    input_tensor = helper.make_tensor_value_info(
        'input', 
        TensorProto.FLOAT, 
        [None]  # Dynamic shape
    )
    
    # Define output
    output_tensor = helper.make_tensor_value_info(
        'output',
        TensorProto.FLOAT,
        [None]
    )
    
    # Create identity node
    node = helper.make_node(
        'Identity',
        inputs=['input'],
        outputs=['output'],
        name='identity_node'
    )
    
    # Create graph
    graph = helper.make_graph(
        [node],
        'identity_graph',
        [input_tensor],
        [output_tensor]
    )
    
    # Create model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 7
    
    # Save
    output_path = os.path.join(output_dir, 'identity.onnx')
    onnx.save(model, output_path)
    print(f"Created: {output_path}")


def create_add_model(output_dir: str) -> None:
    """Create an addition model: c = a + b"""
    
    # Define inputs
    a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [None])
    b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [None])
    
    # Define output
    c = helper.make_tensor_value_info('c', TensorProto.FLOAT, [None])
    
    # Create add node
    node = helper.make_node(
        'Add',
        inputs=['a', 'b'],
        outputs=['c'],
        name='add_node'
    )
    
    # Create graph
    graph = helper.make_graph(
        [node],
        'add_graph',
        [a, b],
        [c]
    )
    
    # Create model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 7
    
    # Save
    output_path = os.path.join(output_dir, 'add.onnx')
    onnx.save(model, output_path)
    print(f"Created: {output_path}")


def create_matmul_model(output_dir: str) -> None:
    """Create a matrix multiplication model: C = A @ B"""
    
    # Define inputs (2D matrices)
    A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
    B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
    
    # Define output
    C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [None, None])
    
    # Create MatMul node
    node = helper.make_node(
        'MatMul',
        inputs=['A', 'B'],
        outputs=['C'],
        name='matmul_node'
    )
    
    # Create graph
    graph = helper.make_graph(
        [node],
        'matmul_graph',
        [A, B],
        [C]
    )
    
    # Create model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 7
    
    # Save
    output_path = os.path.join(output_dir, 'matmul.onnx')
    onnx.save(model, output_path)
    print(f"Created: {output_path}")


def create_relu_model(output_dir: str) -> None:
    """Create a ReLU activation model: output = max(0, input)"""
    
    # Define input
    input_tensor = helper.make_tensor_value_info(
        'input',
        TensorProto.FLOAT,
        [None]
    )
    
    # Define output
    output_tensor = helper.make_tensor_value_info(
        'output',
        TensorProto.FLOAT,
        [None]
    )
    
    # Create ReLU node
    node = helper.make_node(
        'Relu',
        inputs=['input'],
        outputs=['output'],
        name='relu_node'
    )
    
    # Create graph
    graph = helper.make_graph(
        [node],
        'relu_graph',
        [input_tensor],
        [output_tensor]
    )
    
    # Create model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 7
    
    # Save
    output_path = os.path.join(output_dir, 'relu.onnx')
    onnx.save(model, output_path)
    print(f"Created: {output_path}")


def create_identity_int32_model(output_dir: str) -> None:
    """Create an identity model with int32 type"""
    
    input_tensor = helper.make_tensor_value_info(
        'input',
        TensorProto.INT32,
        [None]
    )
    
    output_tensor = helper.make_tensor_value_info(
        'output',
        TensorProto.INT32,
        [None]
    )
    
    node = helper.make_node(
        'Identity',
        inputs=['input'],
        outputs=['output'],
        name='identity_node'
    )
    
    graph = helper.make_graph(
        [node],
        'identity_int32_graph',
        [input_tensor],
        [output_tensor]
    )
    
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 7
    
    output_path = os.path.join(output_dir, 'identity_int32.onnx')
    onnx.save(model, output_path)
    print(f"Created: {output_path}")


def create_simple_neural_network(output_dir: str) -> None:
    """Create a simple neural network with Linear + ReLU + Linear"""
    
    # Define constants
    hidden_size = 64
    input_size = 784  # MNIST-like
    output_size = 10
    
    # Create initializers (weights and biases)
    np.random.seed(42)
    W1 = helper.make_tensor(
        'W1', TensorProto.FLOAT,
        [hidden_size, input_size],
        np.random.randn(hidden_size, input_size).astype(np.float32).flatten().tolist()
    )
    b1 = helper.make_tensor(
        'b1', TensorProto.FLOAT,
        [hidden_size],
        np.random.randn(hidden_size).astype(np.float32).flatten().tolist()
    )
    W2 = helper.make_tensor(
        'W2', TensorProto.FLOAT,
        [output_size, hidden_size],
        np.random.randn(output_size, hidden_size).astype(np.float32).flatten().tolist()
    )
    b2 = helper.make_tensor(
        'b2', TensorProto.FLOAT,
        [output_size],
        np.random.randn(output_size).astype(np.float32).flatten().tolist()
    )
    
    # Define input/output
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [None, input_size])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [None, output_size])
    
    # Create nodes
    # Layer 1: Linear(input @ W1^T + b1)
    transpose1 = helper.make_node('Transpose', ['W1'], ['W1_T'], perm=[1, 0], name='transpose1')
    matmul1 = helper.make_node('MatMul', ['input', 'W1_T'], ['pre_bias1'], name='matmul1')
    add1 = helper.make_node('Add', ['pre_bias1', 'b1'], ['linear1'], name='add1')
    relu = helper.make_node('Relu', ['linear1'], ['hidden'], name='relu')
    
    # Layer 2: Linear(hidden @ W2^T + b2)
    transpose2 = helper.make_node('Transpose', ['W2'], ['W2_T'], perm=[1, 0], name='transpose2')
    matmul2 = helper.make_node('MatMul', ['hidden', 'W2_T'], ['pre_bias2'], name='matmul2')
    add2 = helper.make_node('Add', ['pre_bias2', 'b2'], ['output'], name='add2')
    
    # Create graph
    graph = helper.make_graph(
        [transpose1, matmul1, add1, relu, transpose2, matmul2, add2],
        'simple_nn_graph',
        [input_tensor],
        [output_tensor],
        [W1, b1, W2, b2]
    )
    
    # Create model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 7
    
    # Save
    output_path = os.path.join(output_dir, 'simple_neural_network.onnx')
    onnx.save(model, output_path)
    print(f"Created: {output_path}")


def main():
    # Get output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '..', 'tests', 'Fixtures', 'models')
    
    # Create directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating ONNX test models in: {output_dir}\n")
    
    # Generate models
    create_identity_model(output_dir)
    create_add_model(output_dir)
    create_matmul_model(output_dir)
    create_relu_model(output_dir)
    create_identity_int32_model(output_dir)
    create_simple_neural_network(output_dir)
    
    print("\nAll models generated successfully!")


if __name__ == '__main__':
    main()

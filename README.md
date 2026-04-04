# ONNX Runtime for PHP

[![GitHub Workflow Status (main)](https://img.shields.io/github/actions/workflow/status/phpmlkit/onnxruntime/tests.yml?branch=main&label=tests&style=flat-square)](https://github.com/phpmlkit/onnxruntime/actions)
[![Total Downloads](https://img.shields.io/packagist/dt/phpmlkit/onnxruntime?style=flat-square)](https://packagist.org/packages/phpmlkit/onnxruntime)
[![Latest Version](https://img.shields.io/packagist/v/phpmlkit/onnxruntime?style=flat-square)](https://packagist.org/packages/phpmlkit/onnxruntime)
[![License](https://img.shields.io/github/license/phpmlkit/onnxruntime?style=flat-square)](https://packagist.org/packages/phpmlkit/onnxruntime)

Run machine learning models in PHP using ONNX Runtime. This library provides a complete, type-safe interface to Microsoft's ONNX Runtime through PHP's Foreign Function Interface (FFI).

## What is ONNX Runtime?

ONNX Runtime is a high-performance inference engine for machine learning models. It supports models from PyTorch, TensorFlow, scikit-learn, and many other frameworks that can be converted to the ONNX (Open Neural Network Exchange) format.

This library brings that power to PHP, allowing you to:
- Run pre-trained ML models for image classification, text analysis, recommendations, and more
- Integrate AI capabilities into your PHP applications without external services
- Work with all major ML frameworks through the universal ONNX format

### About This Library

This library is a **reimagined and optimized** version inspired by the original [`onnxruntime-php`](https://github.com/ankane/onnxruntime-php) by Andrew Kane. While the original library provides excellent basic functionality, this version focuses on:

- **FFI-First Architecture**: Direct FFI buffer handling for zero-copy operations with other libraries
- **Comprehensive Type Support**: Full support for sequences, maps, and all ONNX value types
- **First-Class NDArray Support**: Accept NDArray objects as inputs and receive them as outputs for seamless numerical computing
- **Exposed API**: Direct access to `OrtValue` objects for inputs/outputs instead of PHP arrays only

The key difference: this library exposes `OrtValue` objects directly, allowing you to pass data from other FFI libraries without the overhead of copying through PHP arrays. Combined with first-class NDArray support, this enables true zero-copy workflows when working with machine learning pipelines.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Where to Get Models](#where-to-get-models)
- [Core Concepts](#core-concepts)
  - [InferenceSession](#inferencesession)
  - [OrtValue](#ortvalue)
  - [SessionOptions](#sessionoptions)
  - [RunOptions](#runoptions)
- [Working with Data](#working-with-data)
  - [Tensors](#tensors)
  - [Sequences](#sequences)
  - [Maps](#maps)
  - [String Tensors](#string-tensors)
- [Type Support](#type-support)
- [Memory Management](#memory-management)
- [Error Handling](#error-handling)
- [Examples](#examples)
- [Advanced Usage](#advanced-usage)
- [Supported Platforms](#supported-platforms)
- [FFI Direct Access](#ffi-direct-access)
- [Contributing](#contributing)

## Requirements

### PHP Requirements

- **PHP 8.1 or higher**
- **FFI extension enabled**

### Checking FFI Availability

Most PHP installations include FFI but it may be disabled. Check your `php.ini`:

```bash
php -m | grep ffi
```

If FFI is not listed, enable it in your `php.ini`:

```ini
; For PHP 7.4+
extension=ffi

; Ensure FFI is not disabled
ffi.enable=true
```

## Installation

Install via Composer:

```bash
composer require phpmlkit/onnxruntime
```

By default, this installs the `cpu` runtime for your platform.

To use a different runtime, set a runtime override in your application's `composer.json`:

```json
{
  "extra": {
    "platform-packages": {
      "phpmlkit/onnxruntime": {
        "runtime": "cuda12"
      }
    }
  }
}
```

And then reinstall the package to fetch the correct distribution archive:

```bash
composer reinstall phpmlkit/onnxruntime
```

> [!IMPORTANT]
> Run `composer require` or `composer reinstall` on your target platform. Release artifacts include platform-specific native binaries.

### Runtime Variants

| Platform         | Supported Runtimes        |
|------------------|--------------------------|
| Linux x86_64     | `cpu`, `cuda12`, `cuda13`|
| Linux ARM64      | `cpu`                    |
| macOS ARM64      | `cpu`                    |
| Windows x64      | `cpu`, `cuda12`, `cuda13`|

> [!NOTE]
> If your configured runtime is unavailable for your platform, composer will fall back to the `cpu` runtime.

### Manual Library Download

If the native library is missing from your installation, download it manually:

```bash
./vendor/bin/download-onnxruntime
```

Download with specific options:

```bash
./vendor/bin/download-onnxruntime --runtime cuda12
./vendor/bin/download-onnxruntime --runtime cuda13
./vendor/bin/download-onnxruntime --platform windows-x64
./vendor/bin/download-onnxruntime --version 1.24.3
```

Supported script options:
- `--runtime <cpu|cuda12|cuda13>`
- `--platform <linux-x86_64|linux-arm64|darwin-arm64|windows-x64>`
- `--version <onnx-runtime-version>`

**You might need this if:**
- You installed a dev version (branch/tag instead of a release)
- Platform-specific package download failed and composer fell back to source
- You moved `vendor/` directory between different platforms

## Quick Start

Here's a complete example to get you running your first model:

```php
<?php
require_once 'vendor/autoload.php';

use PhpMlKit\ONNXRuntime\InferenceSession;
use PhpMlKit\ONNXRuntime\OrtValue;
use PhpMlKit\ONNXRuntime\Enums\DataType;

$session = InferenceSession::fromFile('/path/to/model.onnx');

$inputData = [1.0, 2.0, 3.0, 4.0, 5.0];
$input = OrtValue::fromArray($inputData, DataType::FLOAT);

$outputs = $session->run(['input' => $input]);

$result = $outputs['output']->toArray();
print_r($result);
```

## Where to Get Models

Wondering where to find ONNX models for the example above? Here are your options:

### 1. Hugging Face Hub (Recommended)

[Hugging Face Hub](https://huggingface.co) is the world's largest collection of machine learning models, including thousands of ONNX-compatible models ready to use. You can browse and filter specifically for ONNX models: https://huggingface.co/models?library=onnx

The easiest way to download these models directly from PHP is using the [Hugging Face PHP](https://github.com/codewithkyrian/huggingface-php) client:

```bash
composer require codewithkyrian/huggingface
```

```php
use Codewithkyrian\HuggingFace\HuggingFace;

$hf = HuggingFace::client();

$modelPath = $hf->hub()
    ->repo('onnx-community/detr-resnet-50-ONNX')
    ->download('onnx/model.onnx');

$session = InferenceSession::fromFile($modelPath);
```

### 2. ONNX Model Zoo (Deprecated)

The official ONNX Model Zoo has been deprecated as of July 2025. Most models previously available there have been migrated to Hugging Face and can be found at:

https://huggingface.co/onnxmodelzoo

### 3. Convert from Other Frameworks

**PyTorch:**
```python
import torch

model = MyModel()
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, 'model.onnx')
```

**TensorFlow/Keras:**
```python
# Use tf2onnx to convert
# pip install tf2onnx
# python -m tf2onnx.convert --saved-model saved_model --output model.onnx
```

**scikit-learn:**
```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('float_input', FloatTensorType([None, 4]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

with open('model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())
```

### 4. Custom Training

Train your own models using any framework (PyTorch, TensorFlow, JAX, etc.) and export to ONNX format.

## Core Concepts

### InferenceSession

The `InferenceSession` is your main interface to ONNX Runtime. It loads models and runs inference.

#### Creating a Session

```php
use PhpMlKit\ONNXRuntime\InferenceSession;

// From file
$session = InferenceSession::fromFile('path/to/model.onnx');

// From bytes
$modelBytes = file_get_contents('model.onnx');
$session = InferenceSession::fromBytes($modelBytes);
```

#### Running Inference

```php
// Basic inference with OrtValue
$input = OrtValue::fromArray([1.0, 2.0, 3.0], DataType::FLOAT);
$outputs = $session->run(['input' => $input]);
$result = $outputs['output']->toArray();

// Get specific outputs only
$outputs = $session->run(
    ['input' => $input],
    ['output1', 'output2']
);

// With run options
$runOptions = RunOptions::default();
$outputs = $session->run(['input' => $input], options: $runOptions);

// With NDArray (requires phpmlkit/ndarray)
$input = NDArray::array([1.0, 2.0, 3.0], DType::Float32);
$outputs = $session->run(['input' => $input]);
$result = $outputs['output']; // NDArray
```

#### Inspecting Model Metadata

```php
// Get input information
$inputs = $session->inputs();
foreach ($inputs as $name => $meta) {
    echo "Input: $name\n";
    echo "  Shape: " . implode(', ', $meta['shape']) . "\n";
    echo "  Type: {$meta['dtype']->name}\n";
}

// Get output information  
$outputs = $session->outputs();
foreach ($outputs as $name => $meta) {
    echo "Output: $name\n";
    echo "  Shape: " . implode(', ', $meta['shape']) . "\n";
    echo "  Type: {$meta['dtype']->name}\n";
}
```

#### Session Lifecycle

Sessions automatically clean up when they go out of scope, but you can explicitly close them:

```php
$session = InferenceSession::fromFile('model.onnx');
// ... use session ...
$session->dispose();  // Explicit cleanup

// Or let PHP handle it automatically when $session goes out of scope
```

> [!IMPORTANT]
> The ONNX environment is shared across all sessions and uses reference counting. It will be automatically cleaned up when the last session closes.

### OrtValue

`OrtValue` is the universal container for all data in ONNX Runtime. It handles:
- **Tensors**: Multi-dimensional arrays of numbers or strings
- **Sequences**: Ordered collections of values
- **Maps**: Key-value pairs
- **Optional**: Optional type wrappers

#### Creating Tensors

```php
use PhpMlKit\ONNXRuntime\OrtValue;
use PhpMlKit\ONNXRuntime\Enums\DataType;

// 1D tensor
$tensor1D = OrtValue::fromArray([1.0, 2.0, 3.0], DataType::FLOAT);

// 2D tensor (matrix)
$tensor2D = OrtValue::fromArray(
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], 
    DataType::FLOAT
);

// 3D tensor
$tensor3D = OrtValue::fromArray(
    [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 
    DataType::INT32
);

// String tensor
$stringTensor = OrtValue::fromArray(
    ['hello', 'world', 'test'], 
    DataType::STRING
);

// With explicit shape
$data = [1, 2, 3, 4, 5, 6];
$tensor = OrtValue::fromArray($data, DataType::INT32, [2, 3]);
```

#### Converting Back to PHP Arrays

```php
// Get data back as PHP array
$result = $tensor->toArray();

// Get tensor information
$shape = $tensor->shape();        // [2, 3]
$type = $tensor->dataType();      // DataType::FLOAT
$count = $tensor->elementCount(); // 6
$bytes = $tensor->sizeInBytes();  // 24 (6 elements × 4 bytes)
```

### SessionOptions

Configure how the session runs with `SessionOptions`:

```php
use PhpMlKit\ONNXRuntime\SessionOptions;
use PhpMlKit\ONNXRuntime\Enums\GraphOptimizationLevel;
use PhpMlKit\ONNXRuntime\Enums\ExecutionMode;

// Method 1: Create with specific options
$options = new SessionOptions(
    graphOptimizationLevel: GraphOptimizationLevel::ENABLE_ALL,
    executionMode: ExecutionMode::PARALLEL,
    interOpNumThreads: 4,
    intraOpNumThreads: 4
);

// Method 2: Use fluent builder
$options = SessionOptions::default()
    ->withGraphOptimizationLevel(GraphOptimizationLevel::ENABLE_ALL)
    ->withExecutionMode(ExecutionMode::PARALLEL)
    ->withInterOpThreads(4)
    ->withIntraOpThreads(4);

// Create session with options
$session = InferenceSession::fromFile('model.onnx', $options);
```

#### Presets

```php
// CPU-optimized preset
$options = SessionOptions::cpuOptimized();

// GPU parallel preset
$options = SessionOptions::gpuParallel();

// Debug preset (verbose logging)
$options = SessionOptions::debug();
```

### RunOptions

Configure individual inference runs:

```php
use PhpMlKit\ONNXRuntime\RunOptions;
use PhpMlKit\ONNXRuntime\Enums\LoggingLevel;

$runOptions = new RunOptions(
    logVerbosityLevel: LoggingLevel::VERBOSE,
    runTag: 'inference_batch_123'
);

// Or use presets
$runOptions = RunOptions::debug();
$runOptions = RunOptions::withTag('my_batch');

// Run with options
$outputs = $session->run($inputs, options: $runOptions);
```

## Working with Data

### Tensors

Tensors are the primary data structure in machine learning. This library supports:

- **Numeric tensors**: FLOAT, DOUBLE, INT8/16/32/64, UINT8/16/32/64
- **String tensors**: Variable-length strings
- **Boolean tensors**: true/false values
- **Multi-dimensional**: 1D, 2D, 3D, and higher dimensions

#### Shape Handling

```php
// Shape is automatically inferred from nested arrays
$tensor = OrtValue::fromArray([[1, 2, 3], [4, 5, 6]], DataType::INT32);
echo $tensor->shape();  // [2, 3]

// Or explicitly specified
$tensor = OrtValue::fromArray([1, 2, 3, 4, 5, 6], DataType::INT32, [2, 3]);
```

#### Dynamic Shapes

Some models accept dynamic shapes (indicated by `-1` in shape):

```php
// Model accepts variable-length input
$meta = $session->inputs()['input'];
echo $meta['shape'];  // Might be [-1] or [-1, 3, 224, 224]

// You can provide any size
$input = OrtValue::fromArray([1, 2, 3], DataType::FLOAT);  // Works
$input = OrtValue::fromArray([1, 2, 3, 4, 5], DataType::FLOAT);  // Also works
```

### Sequences

Sequences are ordered collections of values. Supported element types:
- STRING, INT64, FLOAT, DOUBLE
- Actually all tensor types work (though docs list only those four)

```php
// Create a sequence of tensors
$tensor1 = OrtValue::fromArray([1, 2], DataType::INT32);
$tensor2 = OrtValue::fromArray([3, 4], DataType::INT32);
$tensor3 = OrtValue::fromArray([5, 6], DataType::INT32);

$sequence = OrtValue::sequence([$tensor1, $tensor2, $tensor3]);

// Get sequence length
$length = $sequence->sequenceLength();  // 3

// Access elements
$first = $sequence->getSequenceElement(0);
echo $first->toArray();  // [1, 2]

// Iterate over all elements
$sequence->foreachSequenceElement(function($value, $index) {
    echo "Element $index: " . json_encode($value->toArray()) . "\n";
});

// Convert to PHP array
$result = $sequence->toArray();  // [[1, 2], [3, 4], [5, 6]]
```

### Maps

Maps are key-value pairs with specific type constraints.

#### Supported Map Types

**Key Types:** INT64, STRING  
**Value Types:** INT64, FLOAT, DOUBLE, STRING

```php
use PhpMlKit\ONNXRuntime\OrtValue;
use PhpMlKit\ONNXRuntime\Enums\DataType;

// INT64 keys with FLOAT values
$keys = OrtValue::fromArray([1, 2, 3], DataType::INT64);
$values = OrtValue::fromArray([10.0, 20.0, 30.0], DataType::FLOAT);
$map = OrtValue::map($keys, $values);

$result = $map->toArray();  // [1 => 10.0, 2 => 20.0, 3 => 30.0]

// STRING keys with STRING values
$keys = OrtValue::fromArray(['a', 'b'], DataType::STRING);
$values = OrtValue::fromArray(['x', 'y'], DataType::STRING);
$map = OrtValue::map($keys, $values);

$result = $map->toArray();  // ['a' => 'x', 'b' => 'y']

// Access keys and values separately
$mapKeys = $map->mapKeys();
$mapValues = $map->mapValues();
```

> [!IMPORTANT]
> Other type combinations will throw a `FailException`.

#### Maps in Sequences

ONNX Runtime also supports sequences of maps (specifically for FLOAT values):

```php
// Create maps
$keys1 = OrtValue::fromArray([1, 2], DataType::INT64);
$values1 = OrtValue::fromArray([10.0, 20.0], DataType::FLOAT);
$map1 = OrtValue::map($keys1, $values1);

$keys2 = OrtValue::fromArray([3, 4], DataType::INT64);
$values2 = OrtValue::fromArray([30.0, 40.0], DataType::FLOAT);
$map2 = OrtValue::map($keys2, $values2);

// Create sequence of maps
$sequence = OrtValue::sequence([$map1, $map2]);
$result = $sequence->toArray();  // [[1 => 10.0, 2 => 20.0], [3 => 30.0, 4 => 40.0]]
```

> [!NOTE]
> Sequences of maps only work with INT64/STRING keys and FLOAT values. Other combinations will fail.

### String Tensors

String tensors require special handling due to FFI complexity:

```php
// 1D string tensor
$strings = OrtValue::fromArray(['hello', 'world'], DataType::STRING);

// 2D string tensor
$string2D = OrtValue::fromArray(
    [['a', 'b'], ['c', 'd']], 
    DataType::STRING
);

// Get strings back
$result = $strings->toArray();  // ['hello', 'world']

// Note: Cannot get raw data pointer for strings
// $strings->tensorRawData();  // Throws InvalidOperationException
```

## Type Support

### DataType Enum

All ONNX tensor element types are supported:

| Type | PHP Equivalent | Notes |
|------|---------------|-------|
| `FLOAT` | float | 32-bit floating point |
| `DOUBLE` | float | 64-bit floating point |
| `INT8` | int | 8-bit signed integer |
| `INT16` | int | 16-bit signed integer |
| `INT32` | int | 32-bit signed integer |
| `INT64` | int | 64-bit signed integer |
| `UINT8` | int | 8-bit unsigned integer |
| `UINT16` | int | 16-bit unsigned integer |
| `UINT32` | int | 32-bit unsigned integer |
| `UINT64` | int | 64-bit unsigned integer |
| `BOOL` | bool | Boolean values |
| `STRING` | string | Variable-length strings |

> [!NOTE]
> `FLOAT16`, `BFLOAT16`, `COMPLEX64`, and `COMPLEX128` are defined but may have limited support.

## Memory Management

This library uses a sophisticated memory management system to ensure resources are properly cleaned up while providing flexibility for advanced use cases.

### Automatic Cleanup (RAII)

All resources implement automatic cleanup when they go out of scope:

```php
function processModel() {
    $session = InferenceSession::fromFile('model.onnx');
    $input = OrtValue::fromArray([1, 2, 3], DataType::FLOAT);
    $outputs = $session->run(['input' => $input]);
    
    return $outputs['result']->toArray();
    // $session, $input, $outputs all cleaned up automatically
}
```

### Explicit Cleanup

For deterministic resource management, use explicit cleanup methods:

```php
// Sessions
$session = InferenceSession::fromFile('model.onnx');
// ... use session ...
$session->dispose();  // Release session resources

// OrtValues
$tensor = OrtValue::fromArray([1, 2, 3], DataType::FLOAT);
// ... use tensor ...
$tensor->dispose();  // Release tensor resources

// Safe to call multiple times
$tensor->dispose();  // No error, already disposed
```

### Internal Buffer Management

#### From Array (Internal Buffer)

When you create an OrtValue from a PHP array, the library:
1. Allocates an FFI buffer
2. Copies data from PHP array to FFI buffer
3. Creates ONNX tensor referencing the buffer
4. **Keeps buffer reference to prevent garbage collection**
5. **Automatically releases both on disposal**

```php
$tensor = OrtValue::fromArray([1, 2, 3], DataType::FLOAT);
// Buffer created internally, managed automatically
// Just let $tensor go out of scope or call dispose()
```

#### From Buffer (External Buffer)

When you create an OrtValue from an existing FFI buffer (zero-copy):
1. ONNX tensor references your existing buffer
2. **You are responsible for ensuring buffer outlives the tensor**
3. **You must free the buffer if needed**
4. `dispose()` only releases the tensor handle, not your buffer

```php
$ffi = FFI::cdef();
$buffer = $ffi->new('float[100]');  // Your buffer

$tensor = OrtValue::fromBuffer($buffer, 400, DataType::FLOAT, [100]);
// ... use tensor ...

$tensor->dispose();  // Releases tensor only
// You must manage $buffer lifetime (PHP frees it in this case though!)
```

**Use Case:** External buffers are useful for:
- Zero-copy NDArray integration
- Working with C libraries
- Pre-allocated memory pools

### Session Environment

Sessions share a global ONNX environment with reference counting:

```php
$session1 = InferenceSession::fromFile('model1.onnx');
$session2 = InferenceSession::fromFile('model2.onnx');
// Both share the same environment

$session1->dispose();  // Environment kept alive by session2
$session2->dispose();  // Environment released (no more sessions)
```

This is handled automatically - you don't need to manage it.

## Error Handling

The library provides specific exceptions for different error conditions:

```php
use PhpMlKit\ONNXRuntime\Exceptions\NoSuchFileException;
use PhpMlKit\ONNXRuntime\Exceptions\InvalidProtobufException;
use PhpMlKit\ONNXRuntime\Exceptions\InvalidArgumentException;
use PhpMlKit\ONNXRuntime\Exceptions\FailException;

try {
    $session = InferenceSession::fromFile('model.onnx');
    $outputs = $session->run(['input' => $data]);
} catch (NoSuchFileException $e) {
    // Model file doesn't exist
    echo "Model not found: " . $e->getMessage();
} catch (InvalidProtobufException $e) {
    // File exists but isn't a valid ONNX model
    echo "Invalid model format: " . $e->getMessage();
} catch (InvalidArgumentException $e) {
    // Wrong input name, shape mismatch, etc.
    echo "Invalid input: " . $e->getMessage();
} catch (FailException $e) {
    // General ONNX Runtime error
    echo "ONNX error: " . $e->getMessage();
}
```

### Common Error Codes

| Error | Cause | Solution |
|-------|-------|----------|
| `NoSuchFileException` | Model file not found | Check file path |
| `InvalidProtobufException` | Corrupt or non-ONNX file | Verify model format |
| `InvalidArgumentException` | Wrong input name or shape | Check model metadata |
| `FailException` | General runtime error | Check error message |

## Advanced Usage

### Working with Raw Buffers

For zero-copy operations with other FFI libraries:

```php
$ffi = FFI::cdef();

// Create buffer
$bufferSize = 100 * 4;  // 100 floats × 4 bytes
$buffer = $ffi->new("uint8_t[{$bufferSize}]");

// Fill with data (from another library, file, etc.)
// ... fill $buffer ...

// Create tensor from buffer (zero-copy)
$tensor = OrtValue::fromBuffer(
    $buffer,
    $bufferSize,
    DataType::FLOAT,
    [100]
);

// Use tensor
$outputs = $session->run(['input' => $tensor]);

// Clean up
$tensor->dispose();  // Releases tensor only
// You must free $buffer separately if needed
```

### NDArray Interoperability

When the `phpmlkit/ndarray` package is installed, you can work directly with NDArray objects without manual conversion. Pass NDArrays as inputs and receive NDArrays as outputs:

```php
use PhpMlKit\NDArray\NDArray;
use PhpMlKit\NDArray\DType;

// Create NDArray input
$input = NDArray::array([[1.0, 2.0], [3.0, 4.0]], DType::Float32);

// Run inference - NDArrays in, NDArrays out
$outputs = $session->run(['input' => $input]);

// Get NDArray output directly
$output = $outputs['output'];

echo $output;
// array(2, 2)
// [1. 2.]
// [3. 4.]
```

This provides seamless integration with the NDArray ecosystem for numerical computing in PHP.

### Profiling

Enable profiling to analyze model performance:

```php
use PhpMlKit\ONNXRuntime\SessionOptions;

$options = SessionOptions::default()
    ->withProfiling(true, 'my_model_profile');

$session = InferenceSession::fromFile('model.onnx', $options);

// Run inference multiple times
for ($i = 0; $i < 100; $i++) {
    $session->run($inputs);
}

$session->dispose();  // Profile saved to my_model_profile_*.json
```

### Execution Provider Support

| Provider | Runtime | Status | Platforms |
|----------|---------|--------|-----------|
| **CPUExecutionProvider** | `cpu` | ✅ Included | Linux x86_64/ARM64, macOS ARM64, Windows x64 |
| **CUDAExecutionProvider (CUDA 12)** | `cuda12` | ✅ Available | Linux x86_64, Windows x64 |
| **CUDAExecutionProvider (CUDA 13)** | `cuda13` | ✅ Available | Linux x86_64, Windows x64 |
| **CoreMLExecutionProvider** | `cpu` | ✅ Included | macOS ARM64 |
| **DirectMLExecutionProvider** | N/A | 🚧 Planned | Windows x64 |

**Not supported:**
- **32-bit Systems**: Only 64-bit architectures
- **WebAssembly**: Not supported

To switch runtime variants, update your root `composer.json` and reinstall:

```json
{
  "extra": {
    "platform-packages": {
      "phpmlkit/onnxruntime": {
        "runtime": "cuda13"
      }
    }
  }
}
```

```bash
composer reinstall phpmlkit/onnxruntime
```
The native ONNX Runtime library is automatically bundled for your selected runtime and platform in release artifacts. If you need to support additional platforms, please open an issue.

## FFI Direct Access

For advanced use cases, you can access the underlying FFI layer directly:

```php
use PhpMlKit\ONNXRuntime\FFI\Lib;
use PhpMlKit\ONNXRuntime\FFI\Api;

// Get FFI instance
$ffi = Lib::get();

// Get typed API wrapper
$api = Lib::api();

// Access low-level C API functions
$memoryInfo = $api->createCpuMemoryInfo(
    AllocatorType::ARENA_ALLOCATOR,
    MemoryType::DEFAULT
);

// Don't forget to release resources
$api->releaseMemoryInfo($memoryInfo);
```

**Warning:** Direct FFI access requires knowledge of the ONNX Runtime C API. Use with caution as improper resource management can cause memory leaks or crashes.

### C API Header

The C API header is located at `vendor/phpmlkit/onnxruntime/include/onnxruntime.h`. You can reference it for available functions and types.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
# Clone repository
git clone https://github.com/phpmlkit/onnxruntime.git
cd onnxruntime

# Install dependencies
composer install

# Generate test models (requires Python)
pip install onnx numpy
python scripts/generate_test_models.py

# Run tests
composer test

# Run tests (pretty)
composer test:pretty

# Check code style
composer cs:check

# Fix code style
composer cs:fix

# Run static analysis
composer lint
```

### Code Style

This project follows PSR-12 coding standards. Please run the linter before submitting:

```bash
composer cs:fix
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Microsoft ONNX Runtime](https://github.com/microsoft/onnxruntime) - The underlying inference engine
- [onnxruntime-php](https://github.com/ankane/onnxruntime-php) - The original PHP library that inspired this reimagined version
- [PHP FFI](https://www.php.net/manual/en/book.ffi.php) - Foreign Function Interface
- [Codewithkyrian Platform Package Installer](https://github.com/codewithkyrian/platform-package-installer) - Automatic library distribution

## Support

- **Issues**: https://github.com/phpmlkit/onnxruntime/issues
- **Documentation**: This README and inline PHPDoc
- **Examples**: See `examples/` directory

---

**Happy inferencing! 🚀**
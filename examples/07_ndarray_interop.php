<?php

declare(strict_types=1);

/**
 * Example: NDArray Interoperability with ONNX Runtime.
 */

require_once __DIR__.'/../vendor/autoload.php';

use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\NDArray;
use PhpMlKit\ONNXRuntime\Enums\DataType;
use PhpMlKit\ONNXRuntime\InferenceSession;
use PhpMlKit\ONNXRuntime\OrtValue;

echo "Example: NDArray Interoperability\n";
echo str_repeat('=', 50)."\n\n";

/**
 * Convert NDArray to OrtValue.
 */
function ndarrayToOrtValue(NDArray $ndarray): OrtValue
{
    $shape = $ndarray->shape();
    $dataType = DataType::fromDtype($ndarray->dtype());
    $bufferSize = $ndarray->nbytes();

    $ffi = FFI::cdef();
    $buffer = $ffi->new("uint8_t[{$bufferSize}]");
    $ndarray->intoBuffer($buffer);

    return OrtValue::fromBuffer($buffer, $bufferSize, $dataType, $shape);
}

/**
 * Convert OrtValue to NDArray.
 */
function ortValueToNdArray(OrtValue $ortValue): NDArray
{
    if (!$ortValue->isTensor()) {
        throw new RuntimeException('Cannot convert non-tensor OrtValue to NDArray');
    }

    $dataType = $ortValue->dataType();
    if (DataType::STRING === $dataType) {
        throw new RuntimeException('Cannot convert string tensor to NDArray');
    }

    $dtype = $dataType->toDtype();
    $shape = $ortValue->shape();

    $data = $ortValue->tensorRawData();

    return NDArray::fromBuffer($data, $shape, $dtype);
}

$modelPath = __DIR__.'/../tests/Fixtures/models/identity.onnx';

if (!file_exists($modelPath)) {
    echo "ERROR: Model file not found: {$modelPath}\n";

    exit(1);
}

echo "Loading model...\n";
$session = InferenceSession::fromFile($modelPath);

echo "\n1. Converting NDArray to OrtValue:\n";

// Create NDArray
$ndarray = NDArray::array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::Float32);
echo '   NDArray shape: ['.implode(', ', $ndarray->shape())."]\n";
echo '   NDArray dtype: '.$ndarray->dtype()->name."\n";

// Convert to OrtValue
$input = ndarrayToOrtValue($ndarray);
echo '   OrtValue type: '.$input->dataType()->name."\n";
echo '   OrtValue shape: ['.implode(', ', $input->shape())."]\n";

echo "\n2. Running inference:\n";
$outputs = $session->run(['input' => $input]);

echo "\n3. Converting output back to NDArray:\n";
$outputNdarray = ortValueToNdArray($outputs['output']);
echo '   Output NDArray shape: ['.implode(', ', $outputNdarray->shape())."]\n";
echo '   Output NDArray values: ['.implode(', ', $outputNdarray->toArray())."]\n";

echo "\n4. Summary:\n";
echo "   - NDArray can be easily converted to OrtValue\n";
echo "   - OrtValue can be converted back to NDArray\n";
echo "   - This allows seamless integration with NDArray for advanced operations\n";

echo "\n✓ Example completed successfully!\n";

<?php

declare(strict_types=1);

/**
 * Example: NDArray Interoperability with ONNX Runtime.
 */

require_once __DIR__.'/../vendor/autoload.php';

use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\NDArray;
use PhpMlKit\ONNXRuntime\InferenceSession;
use PhpMlKit\ONNXRuntime\OrtValue;

echo "Example 04: NDArray Interoperability\n";
echo str_repeat('=', 50)."\n\n";

$modelPath = __DIR__.'/../tests/Fixtures/models/identity.onnx';

if (!file_exists($modelPath)) {
    echo "ERROR: Model file not found: {$modelPath}\n";

    exit(1);
}

echo "Loading model...\n";
$session = InferenceSession::fromFile($modelPath);

echo "\n1. Create NDArray input:\n";
$ndarray = NDArray::array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::Float32);
echo '   NDArray shape: ['.implode(', ', $ndarray->shape())."]\n";
echo '   NDArray dtype: '.$ndarray->dtype()->name."\n";

echo "\n2. Convert NDArray -> OrtValue:\n";
$input = OrtValue::fromNDArray($ndarray);
echo '   OrtValue type: '.$input->dataType()->name."\n";
echo '   OrtValue shape: ['.implode(', ', $input->shape())."]\n";

echo "\n3. Run inference:\n";
$outputs = $session->run(['input' => $input]);

echo "\n4. Convert OrtValue -> NDArray:\n";
$outputNdarray = $outputs['output']->toNDArray();
echo '   Output NDArray shape: ['.implode(', ', $outputNdarray->shape())."]\n";
echo '   Output NDArray values: ['.implode(', ', $outputNdarray->toArray())."]\n";

echo "\n✓ Example completed successfully!\n";

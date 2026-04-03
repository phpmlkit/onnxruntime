<?php

declare(strict_types=1);

/**
 * Example 04: NDArray Interoperability with ONNX Runtime.
 *
 * This example demonstrates the first-class NDArray support.
 * You can now pass NDArray objects directly to run() and get NDArray objects back.
 */

require_once __DIR__.'/../vendor/autoload.php';

use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\NDArray;
use PhpMlKit\ONNXRuntime\InferenceSession;

echo "Example 04: NDArray Interoperability\n";
echo str_repeat('=', 50)."\n\n";

$modelPath = __DIR__.'/../tests/Fixtures/models/identity.onnx';

if (!file_exists($modelPath)) {
    echo "ERROR: Model file not found: {$modelPath}\n";

    exit(1);
}

$session = InferenceSession::fromFile($modelPath);

$input = NDArray::array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::Float32);
echo 'Input: '.$input.\PHP_EOL;

$outputs = $session->run(['input' => $input]);

$output = $outputs['output'];
echo 'Output: '.$output.\PHP_EOL;

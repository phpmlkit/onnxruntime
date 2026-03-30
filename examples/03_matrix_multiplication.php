<?php

declare(strict_types=1);

/**
 * Example 3: Matrix Multiplication.
 *
 * Demonstrates 2D tensor operations.
 */

require_once __DIR__.'/../vendor/autoload.php';

use PhpMlKit\ONNXRuntime\Enums\DataType;
use PhpMlKit\ONNXRuntime\InferenceSession;
use PhpMlKit\ONNXRuntime\OrtValue;

echo "Example 3: Matrix Multiplication\n";
echo str_repeat('=', 50)."\n\n";

$modelPath = __DIR__.'/../tests/Fixtures/models/matmul.onnx';

if (!file_exists($modelPath)) {
    echo "ERROR: Model file not found: {$modelPath}\n";

    exit(1);
}

$session = InferenceSession::fromFile($modelPath);

// Create two matrices
$A = OrtValue::fromArray([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0],
], DataType::FLOAT);

$B = OrtValue::fromArray([
    [7.0, 8.0, 9.0],
    [10.0, 11.0, 12.0],
], DataType::FLOAT);

echo 'Matrix A shape: ['.implode(', ', $A->shape())."]\n";
echo "Matrix A data:\n";
foreach ($A->toArray() as $row) {
    echo '  ['.implode(', ', $row)."]\n";
}

echo "\nMatrix B shape: [".implode(', ', $B->shape())."]\n";
echo "Matrix B data:\n";
foreach ($B->toArray() as $row) {
    echo '  ['.implode(', ', $row)."]\n";
}

// Run inference
$outputs = $session->run(['A' => $A, 'B' => $B]);
$C = $outputs['C'];

echo "\nResult C = A @ B shape: [".implode(', ', $C->shape())."]\n";
echo "Result C data:\n";
foreach ($C->toArray() as $row) {
    echo '  ['.implode(', ', $row)."]\n";
}

// Expected result for verification
$expected = [
    [27.0, 30.0, 33.0],
    [61.0, 68.0, 75.0],
    [95.0, 106.0, 117.0],
];

$isCorrect = $C->toArray() === $expected;
echo "\nResult verified: ".($isCorrect ? '✓ PASS' : '✗ FAIL')."\n";

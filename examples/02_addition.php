<?php

declare(strict_types=1);

/**
 * Example 2: Element-wise Addition.
 *
 * Demonstrates running a model with multiple inputs.
 */

require_once __DIR__.'/../vendor/autoload.php';

use PhpMlKit\ONNXRuntime\Enums\DataType;
use PhpMlKit\ONNXRuntime\InferenceSession;
use PhpMlKit\ONNXRuntime\OrtValue;

echo "Example 2: Element-wise Addition\n";
echo str_repeat('=', 50)."\n\n";

$modelPath = __DIR__.'/../tests/Fixtures/models/add.onnx';

if (!file_exists($modelPath)) {
    echo "ERROR: Model file not found: {$modelPath}\n";

    exit(1);
}

$session = InferenceSession::fromFile($modelPath);

$a = OrtValue::fromArray([1.0, 2.0, 3.0, 4.0, 5.0], DataType::FLOAT);
$b = OrtValue::fromArray([10.0, 20.0, 30.0, 40.0, 50.0], DataType::FLOAT);

echo 'Input A: ['.implode(', ', $a->toArray())."]\n";
echo 'Input B: ['.implode(', ', $b->toArray())."]\n\n";

$outputs = $session->run(['a' => $a, 'b' => $b]);
$c = $outputs['c'];

echo 'Output C (A + B): ['.implode(', ', $c->toArray())."]\n\n";

$expected = [11.0, 22.0, 33.0, 44.0, 55.0];
$isCorrect = $c->toArray() === $expected;

echo 'Result verified: '.($isCorrect ? '✓ PASS' : '✗ FAIL')."\n";

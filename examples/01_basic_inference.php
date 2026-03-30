<?php

declare(strict_types=1);

/**
 * Example 1: Basic Inference with Identity Model.
 *
 * This example demonstrates the simplest possible use case:
 * loading a model and running inference.
 */

require_once __DIR__.'/../vendor/autoload.php';

use PhpMlKit\ONNXRuntime\Enums\DataType;
use PhpMlKit\ONNXRuntime\InferenceSession;
use PhpMlKit\ONNXRuntime\OrtValue;

echo "Example 1: Basic Inference with Identity Model\n";
echo str_repeat('=', 50)."\n\n";

$modelPath = __DIR__.'/../tests/Fixtures/models/identity.onnx';

if (!file_exists($modelPath)) {
    echo "ERROR: Model file not found: {$modelPath}\n";
    echo "Please run: python3 scripts/generate_test_models.py\n";

    exit(1);
}

echo "Loading model: identity.onnx\n";
$session = InferenceSession::fromFile($modelPath);

$inputData = [1.0, 2.0, 3.0];
$input = OrtValue::fromArray($inputData, DataType::FLOAT);

echo 'Input shape: ['.implode(', ', $input->shape())."]\n";
echo 'Input data: ['.implode(', ', $inputData)."]\n";
echo "\n";

echo "Running inference...\n";
$outputs = $session->run(['input' => $input]);

$output = $outputs['output'];
$outputData = $output->toArray();
echo 'Output shape: ['.implode(', ', $output->shape())."]\n";
echo 'Output data: ['.implode(', ', $outputData)."]\n";

$isEqual = $inputData === $outputData;
echo "\n";
echo 'Identity verified: '.($isEqual ? '✓ PASS' : '✗ FAIL')."\n";

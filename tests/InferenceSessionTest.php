<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Tests;

use PhpMlKit\NDArray\DType;
use PhpMlKit\NDArray\NDArray;
use PhpMlKit\ONNXRuntime\Enums\DataType;
use PhpMlKit\ONNXRuntime\Enums\ExecutionMode;
use PhpMlKit\ONNXRuntime\Enums\GraphOptimizationLevel;
use PhpMlKit\ONNXRuntime\Enums\OnnxType;
use PhpMlKit\ONNXRuntime\Exception;
use PhpMlKit\ONNXRuntime\Exceptions\FailException;
use PhpMlKit\ONNXRuntime\Exceptions\InvalidArgumentException;
use PhpMlKit\ONNXRuntime\Exceptions\InvalidProtobufException;
use PhpMlKit\ONNXRuntime\Exceptions\NoSuchFileException;
use PhpMlKit\ONNXRuntime\FFI\Lib;
use PhpMlKit\ONNXRuntime\InferenceSession;
use PhpMlKit\ONNXRuntime\Metadata\SequenceMetadata;
use PhpMlKit\ONNXRuntime\Metadata\TensorMetadata;
use PhpMlKit\ONNXRuntime\OrtValue;
use PhpMlKit\ONNXRuntime\SessionOptions;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;

/**
 * @internal
 */
#[CoversClass(InferenceSession::class)]
class InferenceSessionTest extends TestCase
{
    private static string $fixturesDir;

    public static function setUpBeforeClass(): void
    {
        self::$fixturesDir = __DIR__.'/Fixtures/models';
    }

    protected function setUp(): void
    {
        if (!\extension_loaded('ffi')) {
            $this->markTestSkipped('FFI extension not available');
        }

        try {
            Lib::get();
        } catch (Exception $e) {
            $this->markTestSkipped('ONNX Runtime library not available: '.$e->getMessage());
        }
    }

    #[Test]
    public function itCanLoadModelFromFile(): void
    {
        $modelPath = self::$fixturesDir.'/identity.onnx';
        if (!file_exists($modelPath)) {
            $this->markTestSkipped('Test model not available');
        }

        $session = InferenceSession::fromFile($modelPath);
        $this->assertInstanceOf(InferenceSession::class, $session);
    }

    #[Test]
    public function itThrowsForNonexistentModelFile(): void
    {
        $this->expectException(NoSuchFileException::class);
        InferenceSession::fromFile('/nonexistent/path/model.onnx');
    }

    #[Test]
    public function itCanRunIdentityModel(): void
    {
        $modelPath = self::$fixturesDir.'/identity.onnx';
        if (!file_exists($modelPath)) {
            $this->markTestSkipped('Test model not available');
        }

        $session = InferenceSession::fromFile($modelPath);

        $input = OrtValue::fromArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DataType::FLOAT);
        $outputs = $session->run(['input' => $input]);

        $this->assertArrayHasKey('output', $outputs);
        $this->assertEquals([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], $outputs['output']->toArray());
    }

    #[Test]
    public function itCanRunAdditionModel(): void
    {
        $modelPath = self::$fixturesDir.'/add.onnx';
        if (!file_exists($modelPath)) {
            $this->markTestSkipped('Test model not available');
        }

        $session = InferenceSession::fromFile($modelPath);

        $a = OrtValue::fromArray([1.0, 2.0, 3.0], DataType::FLOAT);
        $b = OrtValue::fromArray([4.0, 5.0, 6.0], DataType::FLOAT);
        $outputs = $session->run(['a' => $a, 'b' => $b]);

        $this->assertArrayHasKey('c', $outputs);
        $expected = [5.0, 7.0, 9.0];
        $this->assertEquals($expected, $outputs['c']->toArray());
    }

    #[Test]
    public function itCanRunMatmulModel(): void
    {
        $modelPath = self::$fixturesDir.'/matmul.onnx';
        if (!file_exists($modelPath)) {
            $this->markTestSkipped('Test model not available');
        }

        $session = InferenceSession::fromFile($modelPath);

        $a = OrtValue::fromArray([[1.0, 2.0], [3.0, 4.0]], DataType::FLOAT);
        $b = OrtValue::fromArray([[5.0, 6.0], [7.0, 8.0]], DataType::FLOAT);
        $outputs = $session->run(['A' => $a, 'B' => $b]);

        $this->assertArrayHasKey('C', $outputs);
        // [1*5+2*7, 1*6+2*8] = [19, 22]
        // [3*5+4*7, 3*6+4*8] = [43, 50]
        $expected = [[19.0, 22.0], [43.0, 50.0]];
        $this->assertEquals($expected, $outputs['C']->toArray());
    }

    #[Test]
    public function itCanRunReluModel(): void
    {
        $modelPath = self::$fixturesDir.'/relu.onnx';
        if (!file_exists($modelPath)) {
            $this->markTestSkipped('Test model not available');
        }

        $session = InferenceSession::fromFile($modelPath);

        $input = OrtValue::fromArray([-1.0, 0.0, 1.0, -2.0, 3.0], DataType::FLOAT);
        $outputs = $session->run(['input' => $input]);

        $this->assertArrayHasKey('output', $outputs);
        $expected = [0.0, 0.0, 1.0, 0.0, 3.0];
        $this->assertEquals($expected, $outputs['output']->toArray());
    }

    #[Test]
    public function itSupportsDifferentInputShapes(): void
    {
        $modelPath = self::$fixturesDir.'/identity.onnx';
        if (!file_exists($modelPath)) {
            $this->markTestSkipped('Test model not available');
        }

        $session = InferenceSession::fromFile($modelPath);

        $testCases = [
            [1],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ];

        foreach ($testCases as $shape) {
            $input = OrtValue::fromArray($shape, DataType::FLOAT);
            $outputs = $session->run(['input' => $input]);
            $this->assertEquals($shape, $outputs['output']->toArray());
        }
    }

    #[Test]
    public function itSupportsDifferentDtypes(): void
    {
        $modelPath = self::$fixturesDir.'/identity_int32.onnx';
        if (!file_exists($modelPath)) {
            $this->markTestSkipped('Test model not available');
        }

        $session = InferenceSession::fromFile($modelPath);

        $input = OrtValue::fromArray([1, 2, 3, 4, 5, 6], DataType::INT32);
        $outputs = $session->run(['input' => $input]);

        $this->assertArrayHasKey('output', $outputs);
        $this->assertEquals($input->toArray(), $outputs['output']->toArray());
    }

    #[Test]
    public function itSupportsSessionOptions(): void
    {
        $modelPath = self::$fixturesDir.'/identity.onnx';
        if (!file_exists($modelPath)) {
            $this->markTestSkipped('Test model not available');
        }

        $options = new SessionOptions(
            graphOptimizationLevel: GraphOptimizationLevel::ENABLE_ALL,
            executionMode: ExecutionMode::SEQUENTIAL,
            interOpNumThreads: 1,
            intraOpNumThreads: 1,
        );

        $session = InferenceSession::fromFile($modelPath, $options);
        $this->assertInstanceOf(InferenceSession::class, $session);

        $input = OrtValue::fromArray([1.0, 2.0, 3.0], DataType::FLOAT);
        $outputs = $session->run(['input' => $input]);

        $this->assertEquals([1.0, 2.0, 3.0], $outputs['output']->toArray());
    }

    #[Test]
    public function itCanGetInputInfo(): void
    {
        $modelPath = self::$fixturesDir.'/add.onnx';
        if (!file_exists($modelPath)) {
            $this->markTestSkipped('Test model not available');
        }

        $session = InferenceSession::fromFile($modelPath);
        $inputs = $session->inputs();

        $this->assertCount(2, $inputs);
        $this->assertArrayHasKey('a', $inputs);
        $this->assertArrayHasKey('b', $inputs);

        // Check that inputs are TensorMetadata instances
        $this->assertInstanceOf(TensorMetadata::class, $inputs['a']);
        $this->assertInstanceOf(TensorMetadata::class, $inputs['b']);

        // Check properties
        $this->assertEquals(OnnxType::TENSOR, $inputs['a']->getType());
        $this->assertEquals(DataType::FLOAT, $inputs['a']->getDataType());
        $this->assertEquals([-1], $inputs['a']->getShape());
        $this->assertTrue($inputs['a']->hasDynamicDimensions());

        $this->assertEquals(OnnxType::TENSOR, $inputs['b']->getType());
        $this->assertEquals(DataType::FLOAT, $inputs['b']->getDataType());
        $this->assertEquals([-1], $inputs['b']->getShape());
    }

    #[Test]
    public function itCanGetOutputInfo(): void
    {
        $modelPath = self::$fixturesDir.'/add.onnx';
        if (!file_exists($modelPath)) {
            $this->markTestSkipped('Test model not available');
        }

        $session = InferenceSession::fromFile($modelPath);
        $outputs = $session->outputs();

        $this->assertCount(1, $outputs);
        $this->assertArrayHasKey('c', $outputs);

        $this->assertInstanceOf(TensorMetadata::class, $outputs['c']);

        $this->assertEquals(OnnxType::TENSOR, $outputs['c']->getType());
        $this->assertEquals(DataType::FLOAT, $outputs['c']->getDataType());
        $this->assertEquals([-1], $outputs['c']->getShape());
        $this->assertTrue($outputs['c']->hasDynamicDimensions());
    }

    #[Test]
    public function itCanConvertMetadataToArray(): void
    {
        $modelPath = self::$fixturesDir.'/add.onnx';
        if (!file_exists($modelPath)) {
            $this->markTestSkipped('Test model not available');
        }

        $session = InferenceSession::fromFile($modelPath);
        $inputs = $session->inputs();

        $array = $inputs['a']->toArray();

        $this->assertIsArray($array);
        $this->assertArrayHasKey('type', $array);
        $this->assertArrayHasKey('dataType', $array);
        $this->assertArrayHasKey('shape', $array);
        $this->assertArrayHasKey('symbolicShape', $array);
        $this->assertEquals('TENSOR', $array['type']);
        $this->assertEquals('FLOAT', $array['dataType']);
        $this->assertEquals([-1], $array['shape']);
    }

    #[Test]
    public function itThrowsForInvalidInputName(): void
    {
        $modelPath = self::$fixturesDir.'/identity.onnx';
        if (!file_exists($modelPath)) {
            $this->markTestSkipped('Test model not available');
        }

        $session = InferenceSession::fromFile($modelPath);
        $input = OrtValue::fromArray([1.0], DataType::FLOAT);

        $this->expectException(InvalidArgumentException::class);
        $session->run(['invalid_name' => $input]);
    }

    #[Test]
    public function itThrowsForMissingRequiredInput(): void
    {
        $modelPath = self::$fixturesDir.'/add.onnx';
        if (!file_exists($modelPath)) {
            $this->markTestSkipped('Test model not available');
        }

        $session = InferenceSession::fromFile($modelPath);
        $input = OrtValue::fromArray([1.0], DataType::FLOAT);

        $this->expectException(InvalidArgumentException::class);
        $session->run(['a' => $input]); // Missing 'b'
    }

    #[Test]
    public function itThrowsForWrongInputShape(): void
    {
        $modelPath = self::$fixturesDir.'/matmul.onnx';
        if (!file_exists($modelPath)) {
            $this->markTestSkipped('Test model not available');
        }

        $session = InferenceSession::fromFile($modelPath);

        $a = OrtValue::fromArray([[1.0, 2.0, 3.0]], DataType::FLOAT); // 1x3
        $b = OrtValue::fromArray([[1.0, 2.0]], DataType::FLOAT); // 1x2 - incompatible

        $this->expectException(FailException::class);
        $this->expectExceptionMessage('dimension mismatch');
        $session->run(['A' => $a, 'B' => $b]);
    }

    #[Test]
    public function itCanRunModelWithSpecificOutputs(): void
    {
        $modelPath = self::$fixturesDir.'/identity.onnx';
        if (!file_exists($modelPath)) {
            $this->markTestSkipped('Test model not available');
        }

        $session = InferenceSession::fromFile($modelPath);

        $input = OrtValue::fromArray([1.0, 2.0, 3.0], DataType::FLOAT);
        $outputs = $session->run(['input' => $input], ['output']);

        $this->assertEquals([1.0, 2.0, 3.0], $outputs['output']->toArray());
    }

    #[Test]
    public function itCanBeReusedForMultipleInferences(): void
    {
        $modelPath = self::$fixturesDir.'/identity.onnx';
        if (!file_exists($modelPath)) {
            $this->markTestSkipped('Test model not available');
        }

        $session = InferenceSession::fromFile($modelPath);

        for ($i = 0; $i < 5; ++$i) {
            $input = OrtValue::fromArray([$i, $i + 1, $i + 2], DataType::FLOAT);
            $outputs = $session->run(['input' => $input]);
            $this->assertEquals([$i, $i + 1, $i + 2], $outputs['output']->toArray());
        }
    }

    #[Test]
    public function itHandlesZeroSizeArrayInputs(): void
    {
        $modelPath = self::$fixturesDir.'/identity.onnx';
        if (!file_exists($modelPath)) {
            $this->markTestSkipped('Test model not available');
        }

        $session = InferenceSession::fromFile($modelPath);

        $input = OrtValue::fromArray([], DataType::FLOAT);
        $outputs = $session->run(['input' => $input]);

        $this->assertEquals([], $outputs['output']->toArray());
    }

    #[Test]
    public function itHandlesLargeInputTensors(): void
    {
        $modelPath = self::$fixturesDir.'/identity.onnx';
        if (!file_exists($modelPath)) {
            $this->markTestSkipped('Test model not available');
        }

        $session = InferenceSession::fromFile($modelPath);

        $input = OrtValue::fromArray(array_fill(0, 1000, 1.5), DataType::FLOAT);
        $outputs = $session->run(['input' => $input]);

        $result = $outputs['output']->toArray();
        $this->assertCount(1000, $result);
        $this->assertEquals(1.5, $result[0]);
        $this->assertEquals(1.5, $result[999]);
    }

    #[Test]
    public function memoryIsProperlyReleasedAfterInference(): void
    {
        $modelPath = self::$fixturesDir.'/identity.onnx';
        if (!file_exists($modelPath)) {
            $this->markTestSkipped('Test model not available');
        }

        $session = InferenceSession::fromFile($modelPath);

        for ($i = 0; $i < 100; ++$i) {
            $input = OrtValue::fromArray([$i], DataType::FLOAT);
            $outputs = $session->run(['input' => $input]);

            unset($outputs);
        }

        $this->assertTrue(true); // Test passes if no OOM or errors
    }

    #[Test]
    public function itCanLoadModelFromBytes(): void
    {
        $modelPath = self::$fixturesDir.'/identity.onnx';
        if (!file_exists($modelPath)) {
            $this->markTestSkipped('Test model not available');
        }

        $modelBytes = file_get_contents($modelPath);
        $session = InferenceSession::fromBytes($modelBytes);

        $this->assertInstanceOf(InferenceSession::class, $session);

        $input = OrtValue::fromArray([1.0, 2.0, 3.0], DataType::FLOAT);
        $outputs = $session->run(['input' => $input]);

        $this->assertEquals([1.0, 2.0, 3.0], $outputs['output']->toArray());
    }

    #[Test]
    public function itThrowsInvalidProtobufExceptionForInvalidBytes(): void
    {
        $this->expectException(InvalidProtobufException::class);
        $this->expectExceptionMessage('protobuf parsing failed');

        InferenceSession::fromBytes('this is not valid protobuf content');
    }

    #[Test]
    public function itCanAcceptNDArrayInputs(): void
    {
        if (!class_exists(NDArray::class)) {
            $this->markTestSkipped('NDArray class not available');
        }

        $modelPath = self::$fixturesDir.'/identity.onnx';
        if (!file_exists($modelPath)) {
            $this->markTestSkipped('Test model not available');
        }

        $session = InferenceSession::fromFile($modelPath);

        $input = NDArray::array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::Float32);
        $outputs = $session->run(['input' => $input]);

        $this->assertArrayHasKey('output', $outputs);
        $this->assertInstanceOf(NDArray::class, $outputs['output']);
        $this->assertEquals([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], $outputs['output']->toArray());
    }

    #[Test]
    public function itThrowsForMixedInputTypes(): void
    {
        if (!class_exists(NDArray::class)) {
            $this->markTestSkipped('NDArray class not available');
        }

        $modelPath = self::$fixturesDir.'/add.onnx';
        if (!file_exists($modelPath)) {
            $this->markTestSkipped('Test model not available');
        }

        $session = InferenceSession::fromFile($modelPath);

        $a = OrtValue::fromArray([1.0, 2.0, 3.0], DataType::FLOAT);
        $b = NDArray::array([4.0, 5.0, 6.0], DType::Float32);

        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Mixed input types');
        $session->run(['a' => $a, 'b' => $b]);
    }

    #[Test]
    public function itThrowsForEmptyInputs(): void
    {
        $modelPath = self::$fixturesDir.'/identity.onnx';
        if (!file_exists($modelPath)) {
            $this->markTestSkipped('Test model not available');
        }

        $session = InferenceSession::fromFile($modelPath);

        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Inputs cannot be empty');
        $session->run([]);
    }

    #[Test]
    public function itCanGetSymbolicDimensionsFromImageModel(): void
    {
        $modelPath = self::$fixturesDir.'/image_transform.onnx';
        if (!file_exists($modelPath)) {
            $this->markTestSkipped('Test model not available');
        }

        $session = InferenceSession::fromFile($modelPath);
        $inputs = $session->inputs();

        $this->assertCount(1, $inputs);
        $this->assertArrayHasKey('image', $inputs);

        $imageMetadata = $inputs['image'];
        $this->assertInstanceOf(TensorMetadata::class, $imageMetadata);

        $this->assertEquals([-1, -1, -1, -1], $imageMetadata->getShape());
        $this->assertEquals(['batch_size', 'num_channels', 'height', 'width'], $imageMetadata->getSymbolicShape());

        // Output: channels dimension is constrained to 3 by the bias tensor
        // Bias has shape [3], so output must have exactly 3 channels
        $outputs = $session->outputs();
        $this->assertCount(1, $outputs);
        $outputMetadata = $outputs['output'];
        $this->assertInstanceOf(TensorMetadata::class, $outputMetadata);

        $this->assertEquals([-1, 3, -1, -1], $outputMetadata->getShape());
        $this->assertEquals(['batch_size', '', 'height', 'width'], $outputMetadata->getSymbolicShape());
    }

    #[Test]
    public function itCanGetSequenceMetadata(): void
    {
        $modelPath = self::$fixturesDir.'/sequence_concat.onnx';
        if (!file_exists($modelPath)) {
            $this->markTestSkipped('Test model not available');
        }

        $session = InferenceSession::fromFile($modelPath);
        $inputs = $session->inputs();

        $this->assertCount(1, $inputs);
        $this->assertArrayHasKey('input_sequence', $inputs);

        $sequenceMetadata = $inputs['input_sequence'];
        $this->assertInstanceOf(SequenceMetadata::class, $sequenceMetadata);

        $this->assertEquals(OnnxType::SEQUENCE, $sequenceMetadata->getType());

        $elementMetadata = $sequenceMetadata->getElementMetadata();
        $this->assertInstanceOf(TensorMetadata::class, $elementMetadata);
        $this->assertEquals(OnnxType::TENSOR, $elementMetadata->getType());
        $this->assertEquals(DataType::FLOAT, $elementMetadata->getDataType());

        $outputs = $session->outputs();
        $this->assertCount(1, $outputs);
        $this->assertArrayHasKey('output', $outputs);

        $outputMetadata = $outputs['output'];
        $this->assertInstanceOf(TensorMetadata::class, $outputMetadata);
        $this->assertEquals(OnnxType::TENSOR, $outputMetadata->getType());
        $this->assertEquals(DataType::FLOAT, $outputMetadata->getDataType());
    }
}

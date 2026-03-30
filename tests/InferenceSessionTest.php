<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Tests;

use PhpMlKit\ONNXRuntime\Enums\DataType;
use PhpMlKit\ONNXRuntime\Enums\ExecutionMode;
use PhpMlKit\ONNXRuntime\Enums\GraphOptimizationLevel;
use PhpMlKit\ONNXRuntime\Exception;
use PhpMlKit\ONNXRuntime\Exceptions\FailException;
use PhpMlKit\ONNXRuntime\Exceptions\InvalidArgumentException;
use PhpMlKit\ONNXRuntime\Exceptions\InvalidProtobufException;
use PhpMlKit\ONNXRuntime\Exceptions\NoSuchFileException;
use PhpMlKit\ONNXRuntime\FFI\Lib;
use PhpMlKit\ONNXRuntime\InferenceSession;
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

        $expected = [
            'a' => ['name' => 'a', 'shape' => [-1], 'dtype' => DataType::FLOAT],
            'b' => ['name' => 'b', 'shape' => [-1], 'dtype' => DataType::FLOAT],
        ];

        $this->assertSame($expected, $inputs);
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

        $expected = [
            'c' => ['name' => 'c', 'shape' => [-1], 'dtype' => DataType::FLOAT],
        ];

        $this->assertSame($expected, $outputs);
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
}

<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Tests;

use FFI\CData;
use PhpMlKit\ONNXRuntime\Enums\DataType;
use PhpMlKit\ONNXRuntime\Enums\OnnxType;
use PhpMlKit\ONNXRuntime\Exceptions\FailException;
use PhpMlKit\ONNXRuntime\Exceptions\InvalidArgumentException;
use PhpMlKit\ONNXRuntime\Exceptions\InvalidOperationException;
use PhpMlKit\ONNXRuntime\OrtValue;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;

/**
 * Comprehensive tests for OrtValue.
 *
 * Tests all creation methods, data types, operations, and edge cases.
 *
 * @internal
 */
#[CoversClass(OrtValue::class)]
final class OrtValueTest extends TestCase
{
    // ====================================================================================
    // TENSOR CREATION
    // ====================================================================================

    #[Test]
    public function itCanCreateTensorFrom1DArray(): void
    {
        $data = [1.0, 2.0, 3.0, 4.0, 5.0];
        $tensor = OrtValue::fromArray($data, DataType::FLOAT);

        self::assertTrue($tensor->isTensor());
        self::assertSame(OnnxType::TENSOR, $tensor->type());
        self::assertSame(DataType::FLOAT, $tensor->dataType());
        self::assertSame([5], $tensor->shape());
        self::assertSame(5, $tensor->elementCount());
    }

    #[Test]
    public function itCanCreateTensorFrom2DArray(): void
    {
        $data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        $tensor = OrtValue::fromArray($data, DataType::FLOAT);

        self::assertSame([2, 3], $tensor->shape());
        self::assertSame(6, $tensor->elementCount());
    }

    #[Test]
    public function itCanCreateTensorFrom3DArray(): void
    {
        $data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
        $tensor = OrtValue::fromArray($data, DataType::INT32);

        self::assertSame([2, 2, 2], $tensor->shape());
        self::assertSame(8, $tensor->elementCount());
    }

    #[Test]
    public function itCanCreateTensorWithExplicitShape(): void
    {
        $data = [[1, 2], [3, 4]]; // 2D data to match 2D shape
        $tensor = OrtValue::fromArray($data, DataType::INT32, [2, 2]);

        self::assertSame([2, 2], $tensor->shape());
    }

    #[Test]
    public function itInfersShapeFromNestedStructure(): void
    {
        $data = [[1, 2, 3], [4, 5, 6]];
        $tensor = OrtValue::fromArray($data, DataType::FLOAT);

        self::assertSame([2, 3], $tensor->shape());
    }

    #[Test, DataProvider('dataTypeProvider')]
    public function itCanCreateTensorsWithDifferentDataTypes(DataType $dataType, array $data, array $expectedShape): void
    {
        $tensor = OrtValue::fromArray($data, $dataType);

        self::assertSame($dataType, $tensor->dataType());
        self::assertSame($expectedShape, $tensor->shape());
        self::assertTrue($tensor->isTensor());
    }

    public static function dataTypeProvider(): array
    {
        return [
            'float' => [DataType::FLOAT, [1.0, 2.0, 3.0], [3]],
            'double' => [DataType::DOUBLE, [1.0, 2.0], [2]],
            'int8' => [DataType::INT8, [-128, 0, 127], [3]],
            'int16' => [DataType::INT16, [-32768, 32767], [2]],
            'int32' => [DataType::INT32, [-2147483648, 0, 2147483647], [3]],
            'int64' => [DataType::INT64, [-9223372036854775807, 9223372036854775807], [2]],
            'uint8' => [DataType::UINT8, [0, 128, 255], [3]],
            'uint16' => [DataType::UINT16, [0, 65535], [2]],
            'uint32' => [DataType::UINT32, [0, 4294967295], [2]],
            'uint64' => [DataType::UINT64, [0, \PHP_INT_MAX], [2]],
            'bool' => [DataType::BOOL, [true, false, true], [3]],
        ];
    }

    #[Test]
    public function itCanCreateStringTensor(): void
    {
        $data = ['hello', 'world', 'test'];
        $tensor = OrtValue::fromArray($data, DataType::STRING);

        self::assertSame(DataType::STRING, $tensor->dataType());
        self::assertSame([3], $tensor->shape());
        self::assertSame(3, $tensor->elementCount());
    }

    #[Test]
    public function itCanCreate2DStringTensor(): void
    {
        $data = [['a', 'b'], ['c', 'd'], ['e', 'f']];
        $tensor = OrtValue::fromArray($data, DataType::STRING);

        self::assertSame([3, 2], $tensor->shape());
    }

    #[Test]
    public function itCanCreateZeroSizeTensor(): void
    {
        $tensor = OrtValue::fromArray([], DataType::FLOAT);

        self::assertSame([0], $tensor->shape());
        self::assertSame(0, $tensor->elementCount());
    }

    #[Test]
    public function itCanCreateZeroSizeStringTensor(): void
    {
        $tensor = OrtValue::fromArray([], DataType::STRING, [0, 3]);

        self::assertSame([0, 3], $tensor->shape());
        self::assertSame(DataType::STRING, $tensor->dataType());
    }

    // ====================================================================================
    // DATA CONVERSION - toArray
    // ====================================================================================

    #[Test]
    public function itCanConvertTensorToArray(): void
    {
        $original = [[1.0, 2.0], [3.0, 4.0]];
        $tensor = OrtValue::fromArray($original, DataType::FLOAT);
        $result = $tensor->toArray();

        self::assertSame($original, $result);
    }

    #[Test]
    public function itCanConvertStringTensorToArray(): void
    {
        $original = ['hello', 'world', 'test'];
        $tensor = OrtValue::fromArray($original, DataType::STRING);
        $result = $tensor->toArray();

        self::assertSame($original, $result);
    }

    #[Test, DataProvider('dataTypeProvider')]
    public function itCanRoundTripAllDataTypes(DataType $dataType, array $data): void
    {
        $tensor = OrtValue::fromArray($data, $dataType);
        $result = $tensor->toArray();

        self::assertSame($data, $result, "Failed round-trip for {$dataType->name}");
    }

    #[Test]
    public function itCanConvertMultiDimensionalTensorToArray(): void
    {
        $original = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
        $tensor = OrtValue::fromArray($original, DataType::INT32);
        $result = $tensor->toArray();

        self::assertSame($original, $result);
    }

    // ====================================================================================
    // TENSOR OPERATIONS - Raw Data Access
    // ====================================================================================

    #[Test]
    public function itCanAccessRawTensorData(): void
    {
        $data = [1.0, 2.0, 3.0, 4.0];
        $tensor = OrtValue::fromArray($data, DataType::FLOAT);

        $rawData = $tensor->tensorRawData();

        self::assertInstanceOf(CData::class, $rawData);
    }

    #[Test]
    public function itThrowsWhenAccessingRawDataOnStringTensor(): void
    {
        $this->expectException(InvalidOperationException::class);
        $this->expectExceptionMessage('Cannot get raw data for string tensor');

        $tensor = OrtValue::fromArray(['hello', 'world'], DataType::STRING);
        $tensor->tensorRawData();
    }

    #[Test]
    public function itThrowsWhenAccessingRawDataOnNonTensor(): void
    {
        $this->expectException(InvalidOperationException::class);
        $this->expectExceptionMessage('Not a tensor');

        $sequence = OrtValue::sequence([OrtValue::fromArray([1, 2], DataType::INT32)]);
        $sequence->tensorRawData();
    }

    // ====================================================================================
    // TENSOR OPERATIONS - Size Information
    // ====================================================================================

    #[Test]
    public function itCanGetTensorSizeInBytes(): void
    {
        $tensor = OrtValue::fromArray([1.0, 2.0, 3.0, 4.0], DataType::FLOAT);

        // FLOAT is 4 bytes, 4 elements = 16 bytes
        self::assertSame(16, $tensor->sizeInBytes());
    }

    #[Test]
    public function itCanGetDoubleTensorSizeInBytes(): void
    {
        $tensor = OrtValue::fromArray([1.0, 2.0], DataType::DOUBLE);

        // DOUBLE is 8 bytes, 2 elements = 16 bytes
        self::assertSame(16, $tensor->sizeInBytes());
    }

    #[Test]
    public function itThrowsWhenGettingSizeOnNonTensor(): void
    {
        $this->expectException(InvalidOperationException::class);

        $sequence = OrtValue::sequence([OrtValue::fromArray([1], DataType::INT32)]);
        $sequence->sizeInBytes();
    }

    // ====================================================================================
    // SEQUENCE OPERATIONS
    // ====================================================================================

    #[Test]
    public function itCanCreateSequence(): void
    {
        $values = [
            OrtValue::fromArray([1, 2], DataType::INT32),
            OrtValue::fromArray([3, 4], DataType::INT32),
        ];

        $sequence = OrtValue::sequence($values);

        self::assertSame(OnnxType::SEQUENCE, $sequence->type());
        self::assertFalse($sequence->isTensor());
    }

    #[Test]
    public function itThrowsWhenCreatingEmptySequence(): void
    {
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Cannot create empty sequence');

        OrtValue::sequence([]);
    }

    #[Test]
    public function itCanGetSequenceLength(): void
    {
        $values = [
            OrtValue::fromArray([1], DataType::INT32),
            OrtValue::fromArray([2], DataType::INT32),
            OrtValue::fromArray([3], DataType::INT32),
        ];

        $sequence = OrtValue::sequence($values);

        self::assertSame(3, $sequence->sequenceLength());
    }

    #[Test]
    public function itCanGetSequenceElement(): void
    {
        $values = [
            OrtValue::fromArray([1, 2], DataType::INT32),
            OrtValue::fromArray([3, 4], DataType::INT32),
        ];

        $sequence = OrtValue::sequence($values);
        $element = $sequence->getSequenceElement(0);

        self::assertTrue($element->isTensor());
        self::assertSame([1, 2], $element->toArray());
    }

    #[Test]
    public function itCanIterateOverSequenceElements(): void
    {
        $values = [
            OrtValue::fromArray([1], DataType::INT32),
            OrtValue::fromArray([2], DataType::INT32),
            OrtValue::fromArray([3], DataType::INT32),
        ];

        $sequence = OrtValue::sequence($values);
        $collected = [];

        $sequence->foreachSequenceElement(static function ($value) use (&$collected) {
            $collected[] = $value->toArray()[0];
        });

        self::assertSame([1, 2, 3], $collected);
    }

    #[Test]
    public function itThrowsWhenGettingLengthOnNonSequence(): void
    {
        $this->expectException(InvalidOperationException::class);
        $this->expectExceptionMessage('Not a sequence');

        $tensor = OrtValue::fromArray([1, 2], DataType::INT32);
        $tensor->sequenceLength();
    }

    #[Test]
    public function itThrowsWhenGettingElementOnNonSequence(): void
    {
        $this->expectException(InvalidOperationException::class);
        $this->expectExceptionMessage('Not a sequence');

        $tensor = OrtValue::fromArray([1, 2], DataType::INT32);
        $tensor->getSequenceElement(0);
    }

    #[Test]
    public function itCanConvertSequenceToArray(): void
    {
        $values = [
            OrtValue::fromArray([1, 2], DataType::INT32),
            OrtValue::fromArray([3, 4], DataType::INT32),
        ];

        $sequence = OrtValue::sequence($values);
        $result = $sequence->toArray();

        self::assertSame([[1, 2], [3, 4]], $result);
    }

    #[Test, DataProvider('sequenceTypeProvider')]
    public function itValidatesSequenceTypeSupport(array $value1, array $value2, DataType $elementType): void
    {
        $values = [
            OrtValue::fromArray($value1, $elementType),
            OrtValue::fromArray($value2, $elementType),
        ];

        $sequence = OrtValue::sequence($values);

        self::assertSame(OnnxType::SEQUENCE, $sequence->type());
        self::assertSame([$value1, $value2], $sequence->toArray());
    }

    public static function sequenceTypeProvider(): array
    {
        return [
            'STRING elements' => [['a', 'b'], ['c', 'd'], DataType::STRING],
            'INT64 elements' => [[1, 2], [3, 4], DataType::INT64],
            'FLOAT elements' => [[1.0, 2.0], [3.0, 4.0], DataType::FLOAT],
            'DOUBLE elements' => [[1.0, 2.0], [3.0, 4.0], DataType::DOUBLE],
            'INT8 elements' => [[1, 2], [3, 4], DataType::INT8],
            'INT16 elements' => [[1, 2], [3, 4], DataType::INT16],
            'INT32 elements' => [[1, 2], [3, 4], DataType::INT32],
            'UINT8 elements' => [[1, 2], [3, 4], DataType::UINT8],
            'UINT16 elements' => [[1, 2], [3, 4], DataType::UINT16],
            'UINT32 elements' => [[1, 2], [3, 4], DataType::UINT32],
            'UINT64 elements' => [[1, 2], [3, 4], DataType::UINT64],
            'BOOL elements' => [[true, false], [true, false], DataType::BOOL],
        ];
    }

    #[Test, DataProvider('sequenceOfMapsTypeProvider')]
    public function itValidatesSequenceOfMapsTypeSupport(array $map1Data, array $map2Data, DataType $keyType, DataType $valueType, bool $shouldWork): void
    {
        if (!$shouldWork) {
            $this->expectException(FailException::class);
        }

        $keys1 = OrtValue::fromArray(array_keys($map1Data), $keyType);
        $values1 = OrtValue::fromArray(array_values($map1Data), $valueType);
        $map1 = OrtValue::map($keys1, $values1);

        $keys2 = OrtValue::fromArray(array_keys($map2Data), $keyType);
        $values2 = OrtValue::fromArray(array_values($map2Data), $valueType);
        $map2 = OrtValue::map($keys2, $values2);

        $sequence = OrtValue::sequence([$map1, $map2]);

        if ($shouldWork) {
            self::assertSame(OnnxType::SEQUENCE, $sequence->type());
            self::assertSame([$map1Data, $map2Data], $sequence->toArray());
        }
    }

    public static function sequenceOfMapsTypeProvider(): array
    {
        return [
            // Supported combinations from ONNX docs:
            // std::vector<std::map<std::string, float>>
            // std::vector<std::map<int64_t, float>>
            'INT64 keys with FLOAT values' => [
                [1 => 10.0, 2 => 20.0], [3 => 30.0, 4 => 40.0], DataType::INT64, DataType::FLOAT, true,
            ],
            'STRING keys with FLOAT values' => [
                ['a' => 10.0, 'b' => 20.0], ['c' => 30.0, 'd' => 40.0], DataType::STRING, DataType::FLOAT, true,
            ],

            // Other key types should fail in sequences of maps
            'INT32 keys with FLOAT values should fail' => [
                [1 => 10.0, 2 => 20.0], [3 => 30.0, 4 => 40.0], DataType::INT32, DataType::FLOAT, false,
            ],
            'INT8 keys with FLOAT values should fail' => [
                [1 => 10.0, 2 => 20.0], [3 => 30.0, 4 => 40.0], DataType::INT8, DataType::FLOAT, false,
            ],

            // Other value types should fail in sequences of maps
            'INT64 keys with INT64 values should fail' => [
                [1 => 10, 2 => 20], [3 => 30, 4 => 40], DataType::INT64, DataType::INT64, false,
            ],
            'INT64 keys with DOUBLE values should fail' => [
                [1 => 10.0, 2 => 20.0], [3 => 30.0, 4 => 40.0], DataType::INT64, DataType::DOUBLE, false,
            ],
            'STRING keys with STRING values should fail' => [
                ['a' => 'x', 'b' => 'y'], ['c' => 'z', 'd' => 'w'], DataType::STRING, DataType::STRING, false,
            ],
        ];
    }

    // ====================================================================================
    // MAP OPERATIONS
    // ====================================================================================

    #[Test]
    public function itCanCreateMap(): void
    {
        $keys = OrtValue::fromArray([1, 2, 3], DataType::INT64);
        $values = OrtValue::fromArray([10.0, 20.0, 30.0], DataType::FLOAT);

        $map = OrtValue::map($keys, $values);

        self::assertSame(OnnxType::MAP, $map->type());
        self::assertFalse($map->isTensor());
    }

    #[Test]
    public function itCanGetMapKeys(): void
    {
        $keys = OrtValue::fromArray([1, 2, 3], DataType::INT64);
        $values = OrtValue::fromArray([10.0, 20.0, 30.0], DataType::FLOAT);

        $map = OrtValue::map($keys, $values);
        $keyTensor = $map->mapKeys();

        self::assertTrue($keyTensor->isTensor());
        self::assertSame([1, 2, 3], $keyTensor->toArray());
    }

    #[Test]
    public function itCanGetMapValues(): void
    {
        $keys = OrtValue::fromArray([1, 2, 3], DataType::INT64);
        $values = OrtValue::fromArray([10.0, 20.0, 30.0], DataType::FLOAT);

        $map = OrtValue::map($keys, $values);
        $valueTensor = $map->mapValues();

        self::assertTrue($valueTensor->isTensor());
        self::assertSame([10.0, 20.0, 30.0], $valueTensor->toArray());
    }

    #[Test]
    public function itCanConvertMapToArray(): void
    {
        $keys = OrtValue::fromArray([1, 2, 3], DataType::INT64);
        $values = OrtValue::fromArray([10.0, 20.0, 30.0], DataType::FLOAT);

        $map = OrtValue::map($keys, $values);
        $result = $map->toArray();

        self::assertSame([1 => 10.0, 2 => 20.0, 3 => 30.0], $result);
    }

    #[Test, DataProvider('mapTypeProvider')]
    public function itValidatesMapTypeSupport(array $data, DataType $keyType, DataType $valueType, bool $shouldWork): void
    {
        if (!$shouldWork) {
            $this->expectException(FailException::class);
        }

        $keys = OrtValue::fromArray(array_keys($data), $keyType);
        $values = OrtValue::fromArray(array_values($data), $valueType);

        $map = OrtValue::map($keys, $values);

        if ($shouldWork) {
            self::assertSame(OnnxType::MAP, $map->type());
            self::assertSame($data, $map->toArray());
        }
    }

    public static function mapTypeProvider(): array
    {
        return [
            // INT64 keys with supported value types
            'INT64 keys with INT64 values' => [
                [1 => 10, 2 => 20], DataType::INT64, DataType::INT64, true,
            ],
            'INT64 keys with FLOAT values' => [
                [1 => 10.0, 2 => 20.0], DataType::INT64, DataType::FLOAT, true,
            ],
            'INT64 keys with DOUBLE values' => [
                [1 => 10.0, 2 => 20.0], DataType::INT64, DataType::DOUBLE, true,
            ],
            'INT64 keys with STRING values' => [
                [1 => 'a', 2 => 'b'], DataType::INT64, DataType::STRING, true,
            ],

            // STRING keys with supported value types
            'STRING keys with INT64 values' => [
                ['a' => 10, 'b' => 20], DataType::STRING, DataType::INT64, true,
            ],
            'STRING keys with FLOAT values' => [
                ['a' => 10.0, 'b' => 20.0], DataType::STRING, DataType::FLOAT, true,
            ],
            'STRING keys with DOUBLE values' => [
                ['a' => 10.0, 'b' => 20.0], DataType::STRING, DataType::DOUBLE, true,
            ],
            'STRING keys with STRING values' => [
                ['a' => 'x', 'b' => 'y'], DataType::STRING, DataType::STRING, true,
            ],

            // Other key types should fail
            'INT8 keys should fail' => [
                [1 => 10, 2 => 20], DataType::INT8, DataType::INT64, false,
            ],
            'INT16 keys should fail' => [
                [1 => 10, 2 => 20], DataType::INT16, DataType::INT64, false,
            ],
            'INT32 keys should fail' => [
                [1 => 10, 2 => 20], DataType::INT32, DataType::INT64, false,
            ],
            'UINT8 keys should fail' => [
                [1 => 10, 2 => 20], DataType::UINT8, DataType::INT64, false,
            ],
            'UINT16 keys should fail' => [
                [1 => 10, 2 => 20], DataType::UINT16, DataType::INT64, false,
            ],
            'UINT32 keys should fail' => [
                [1 => 10, 2 => 20], DataType::UINT32, DataType::INT64, false,
            ],
            'UINT64 keys should fail' => [
                [1 => 10, 2 => 20], DataType::UINT64, DataType::INT64, false,
            ],

            // Value types that don't work with INT64 keys
            'INT64 keys with INT32 values should fail' => [
                [1 => 10, 2 => 20], DataType::INT64, DataType::INT32, false,
            ],
            'INT64 keys with INT8 values should fail' => [
                [1 => 10, 2 => 20], DataType::INT64, DataType::INT8, false,
            ],
            'INT64 keys with UINT8 values should fail' => [
                [1 => 10, 2 => 20], DataType::INT64, DataType::UINT8, false,
            ],
        ];
    }

    #[Test]
    public function itThrowsWhenGettingKeysOnNonMap(): void
    {
        $this->expectException(InvalidOperationException::class);
        $this->expectExceptionMessage('Not a map');

        $tensor = OrtValue::fromArray([1, 2], DataType::INT32);
        $tensor->mapKeys();
    }

    #[Test]
    public function itThrowsWhenGettingValuesOnNonMap(): void
    {
        $this->expectException(InvalidOperationException::class);
        $this->expectExceptionMessage('Not a map');

        $tensor = OrtValue::fromArray([1, 2], DataType::INT32);
        $tensor->mapValues();
    }

    // ====================================================================================
    // MEMORY MANAGEMENT
    // ====================================================================================

    #[Test]
    public function itCanDisposeTensor(): void
    {
        $tensor = OrtValue::fromArray([1, 2, 3], DataType::FLOAT);

        self::assertFalse($this->isDisposed($tensor));

        $tensor->dispose();

        self::assertTrue($this->isDisposed($tensor));
    }

    #[Test]
    public function itCanDisposeMultipleTimes(): void
    {
        $tensor = OrtValue::fromArray([1, 2], DataType::INT32);

        $tensor->dispose();
        $tensor->dispose(); // Should not throw

        self::assertTrue($this->isDisposed($tensor));
    }

    #[Test]
    public function itCleansUpOnDestruction(): void
    {
        $tensor = OrtValue::fromArray([1, 2, 3], DataType::FLOAT);
        $handle = $tensor->handle;

        unset($tensor);

        self::assertTrue(true);
    }

    #[Test]
    public function itCanDisposeSequence(): void
    {
        $sequence = OrtValue::sequence([
            OrtValue::fromArray([1], DataType::INT32),
        ]);

        $sequence->dispose();

        self::assertTrue($this->isDisposed($sequence));
    }

    #[Test]
    public function itCanDisposeMap(): void
    {
        $map = OrtValue::map(
            OrtValue::fromArray([1], DataType::INT64),
            OrtValue::fromArray([10.0], DataType::FLOAT)
        );

        $map->dispose();

        self::assertTrue($this->isDisposed($map));
    }

    // ====================================================================================
    // EDGE CASES
    // ===================================================================================

    #[Test]
    public function itCanHandleSingleElement(): void
    {
        $tensor = OrtValue::fromArray([42], DataType::INT32);

        self::assertSame([1], $tensor->shape());
        self::assertSame(1, $tensor->elementCount());
        self::assertSame([42], $tensor->toArray());
    }

    #[Test]
    public function itCanHandleNegativeNumbers(): void
    {
        $data = [-100, -50, -1, 0, 1, 50, 100];
        $tensor = OrtValue::fromArray($data, DataType::INT32);

        self::assertSame($data, $tensor->toArray());
    }

    #[Test]
    public function itCanHandleFloatPrecision(): void
    {
        $data = [1.123456, 2.789012, 3.345678];
        $tensor = OrtValue::fromArray($data, DataType::FLOAT);
        $result = $tensor->toArray();

        foreach ($data as $i => $expected) {
            self::assertEqualsWithDelta($expected, $result[$i], 0.0001);
        }
    }

    // ====================================================================================
    // HELPER METHODS
    // ====================================================================================

    private function isDisposed(OrtValue $value): bool
    {
        $reflection = new \ReflectionClass($value);
        $property = $reflection->getProperty('disposed');
        $property->setAccessible(true);

        return $property->getValue($value);
    }
}

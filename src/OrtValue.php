<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime;

use FFI\CData;
use PhpMlKit\NDArray\NDArray;
use PhpMlKit\ONNXRuntime\Contracts\Disposable;
use PhpMlKit\ONNXRuntime\Enums\AllocatorType;
use PhpMlKit\ONNXRuntime\Enums\DataType;
use PhpMlKit\ONNXRuntime\Enums\MemoryType;
use PhpMlKit\ONNXRuntime\Enums\OnnxType;
use PhpMlKit\ONNXRuntime\Exceptions\InvalidArgumentException;
use PhpMlKit\ONNXRuntime\Exceptions\InvalidOperationException;
use PhpMlKit\ONNXRuntime\FFI\Lib;

/**
 * Represents an ONNX Runtime value.
 *
 * This is the primary data container for all ONNX types:
 * - Tensors (all element types including strings)
 * - Sequences
 * - Maps
 * - Sparse tensors
 * - Opaque types
 * - Optional types
 */
final class OrtValue implements Disposable
{
    private bool $disposed = false;

    /**
     * Private constructor. Use factory methods.
     *
     * @param CData      $handle       Native OrtValue handle
     * @param OnnxType   $type         ONNX type
     * @param null|CData $memoryHandle Optional internal buffer reference (keeps memory alive)
     * @param DataType   $dataType     Data type (Undefined if not tensor)
     * @param null|array $shape        Shape (if tensor)
     */
    private function __construct(
        public readonly CData $handle,
        private OnnxType $type,
        private DataType $dataType,
        private ?array $shape = null,
        private ?CData $memoryHandle = null,
    ) {}

    public function __destruct()
    {
        $this->dispose();
    }

    /**
     * Dispose of native resources.
     *
     * Safe to call multiple times.
     */
    public function dispose(): void
    {
        if ($this->disposed) {
            return;
        }

        $api = Lib::api();

        $api->releaseValue($this->handle);

        $this->memoryHandle = null;

        $this->disposed = true;
    }

    /**
     * Get the ONNX value type.
     */
    public function type(): OnnxType
    {
        return $this->type;
    }

    /**
     * Check if this OrtValue contains a tensor (dense or sparse).
     */
    public function isTensor(): bool
    {
        return $this->type()->isTensor();
    }

    /**
     * Get tensor element type.
     */
    public function dataType(): ?DataType
    {
        return $this->dataType;
    }

    /**
     * Get tensor shape.
     *
     * @return null|int[]
     */
    public function shape(): ?array
    {
        return $this->shape;
    }

    /**
     * Get number of elements in tensor.
     */
    public function elementCount(): ?int
    {
        $shape = $this->shape();

        return array_product($shape) ?: 0;
    }

    /**
     * Get tensor size in bytes.
     *
     * @throws InvalidOperationException If not a tensor
     */
    public function sizeInBytes(): int
    {
        if (!$this->isTensor()) {
            throw new InvalidOperationException('Not a tensor: '.$this->type()->getName());
        }

        if (DataType::STRING === $this->dataType) {
            throw new InvalidOperationException('Cannot get size in bytes for string tensor');
        }

        $elementCount = $this->elementCount();

        return $elementCount * $this->dataType->sizeInBytes();
    }

    /**
     * Create tensor from PHP array.
     *
     * This method creates an internal FFI buffer, copies the data, and creates
     * an OrtValue. The buffer is managed by the OrtValue and released on disposal.
     *
     * @param array      $data       Nested PHP array (e.g., [[1,2], [3,4]] for 2x2 tensor)
     * @param DataType   $dataType   Element type
     * @param null|array $shape      Optional shape. If null, inferred from array structure
     * @param null|CData $memoryInfo Optional memory info. Uses CPU memory if null.
     *
     * @throws InvalidArgumentException If data doesn't match shape
     */
    public static function fromArray(array $data, DataType $dataType, ?array $shape = null, ?CData $memoryInfo = null): self
    {
        $ffi = Lib::get();

        $shape ??= self::inferShape($data);

        if (DataType::STRING === $dataType) {
            return self::createStringTensorFromArray($data, $shape);
        }

        $flatData = self::flattenArray($data, $shape);
        $size = array_product($shape);

        if (\count($flatData) !== $size) {
            throw new InvalidArgumentException(
                'Data size ('.\count($flatData).") doesn't match shape size ({$size})"
            );
        }

        $ffiType = $dataType->ffiType();
        $buffer = $size > 0 ? $ffi->new("{$ffiType}[{$size}]") : $ffi->new('void *');
        $bufferSize = $size * $dataType->sizeInBytes();

        foreach ($flatData as $i => $value) {
            $buffer[$i] = $value;
        }

        return self::fromBuffer($buffer, $bufferSize, $dataType, $shape, $memoryInfo);
    }

    /**
     * Create tensor from FFI buffer (zero-copy).
     *
     * WARNING: The user is responsible for ensuring the buffer outlives the OrtValue.
     * The buffer will NOT be freed when the OrtValue is destroyed.
     *
     * @param CData      $buffer     FFI buffer pointer
     * @param int        $bufferSize Buffer size in bytes
     * @param DataType   $dataType   Tensor element type
     * @param array      $shape      Tensor shape
     * @param null|CData $memoryInfo Optional memory info
     */
    public static function fromBuffer(CData $buffer, int $bufferSize, DataType $dataType, array $shape, ?CData $memoryInfo = null): self
    {
        $api = Lib::api();

        $cleanupMemoryInfo = null === $memoryInfo;
        $memoryInfo ??= $api->createCpuMemoryInfo(AllocatorType::ARENA_ALLOCATOR, MemoryType::DEFAULT);

        try {
            $handle = $api->createTensorWithDataAsOrtValue($memoryInfo, $buffer, $bufferSize, $shape, $dataType);

            return new self($handle, OnnxType::TENSOR, $dataType, $shape, $buffer);
        } finally {
            if ($cleanupMemoryInfo) {
                $api->releaseMemoryInfo($memoryInfo);
            }
        }
    }

    /**
     * Create tensor from NDArray.
     *
     * @throws InvalidArgumentException If NDArray package is not installed
     */
    public static function fromNDArray(NDArray $ndarray): self
    {
        if (!class_exists(NDArray::class)) {
            throw new InvalidArgumentException('NDArray support requires phpmlkit/ndarray. Install the package to use fromNDArray().');
        }

        $shape = $ndarray->shape();
        $dataType = DataType::fromDtype($ndarray->dtype());
        $bufferSize = $ndarray->nbytes();

        $buffer = Lib::get()->new("uint8_t[{$bufferSize}]");
        $ndarray->intoBuffer($buffer);

        return self::fromBuffer($buffer, $bufferSize, $dataType, $shape);
    }

    /**
     * Create from native handle.
     *
     * This is used internally when receiving values from the API.
     * Type info is extracted once and cached.
     *
     * @param CData $handle Native OrtValue handle
     */
    public static function fromHandle(CData $handle): self
    {
        $api = Lib::api();
        $type = OnnxType::from($api->getValueType($handle));

        $dataType = DataType::UNDEFINED;
        $shape = null;

        if ($type->isTensor()) {
            $tensorInfo = $api->getTensorTypeAndShape($handle);

            try {
                $dataType = $api->getTensorElementType($tensorInfo);
                $shape = $api->getDimensions($tensorInfo);
            } finally {
                $api->releaseTensorTypeAndShapeInfo($tensorInfo);
            }
        }

        return new self($handle, $type, $dataType, $shape);
    }

    /**
     * Create sequence from array of OrtValues.
     *
     * @param array<self> $values Array of OrtValues
     *
     * @throws InvalidArgumentException If values array is empty
     */
    public static function sequence(array $values): self
    {
        if (empty($values)) {
            throw new InvalidArgumentException('Cannot create empty sequence');
        }

        $api = Lib::api();

        $handles = [];
        foreach ($values as $v) {
            if (!$v instanceof self) {
                throw new InvalidArgumentException('All elements must be instances of OrtValue');
            }
            $handles[] = $v->handle;
        }
        $handle = $api->createValue($handles, OnnxType::SEQUENCE);

        return new self($handle, OnnxType::SEQUENCE, DataType::UNDEFINED);
    }

    /**
     * Create map from keys and values OrtValues.
     *
     * @param self $keys   Keys tensor (must be primitive type)
     * @param self $values Values tensor or sequence
     */
    public static function map(self $keys, self $values): self
    {
        $api = Lib::api();
        $handle = $api->CreateValue([$keys->handle, $values->handle], OnnxType::MAP);

        return new self($handle, OnnxType::MAP, DataType::UNDEFINED);
    }

    /**
     * Convert OrtValue to PHP data structure.
     *
     * - Tensors: Returns nested PHP array
     * - Sequences: Returns array of converted values
     * - Maps: Returns associative array
     * - String tensors: Returns nested array of strings
     *
     * @return array|mixed
     *
     * @throws InvalidOperationException For unsupported types
     */
    public function toArray(): array
    {
        return match ($this->type) {
            OnnxType::TENSOR, OnnxType::SPARSE_TENSOR => $this->tensorToArray(),
            OnnxType::SEQUENCE => $this->sequenceToArray(),
            OnnxType::MAP => $this->mapToArray(),
            default => throw new InvalidOperationException(
                "Cannot convert {$this->type->getName()} to array"
            ),
        };
    }

    /**
     * Convert tensor OrtValue to NDArray.
     *
     * @throws InvalidArgumentException  If NDArray package is not installed
     * @throws InvalidOperationException If value is not a supported tensor
     */
    public function toNDArray(): NDArray
    {
        if (!class_exists(NDArray::class)) {
            throw new InvalidArgumentException('NDArray support requires phpmlkit/ndarray. Install the package to use toNDArray().');
        }

        if (!$this->isTensor()) {
            throw new InvalidOperationException('Cannot convert non-tensor OrtValue to NDArray');
        }

        if (DataType::STRING === $this->dataType) {
            throw new InvalidOperationException('Cannot convert string tensor to NDArray');
        }

        $dtype = $this->dataType->toDtype();
        $shape = $this->shape();
        $data = $this->tensorRawData();

        return NDArray::fromBuffer($data, $shape, $dtype);
    }

    /**
     * Get raw tensor data buffer.
     *
     * Provides direct access to native memory. Only valid for non-string tensors.
     *
     * @return CData Pointer to tensor data
     *
     * @throws InvalidOperationException For string tensors
     */
    public function tensorRawData(): CData
    {
        if (!$this->isTensor()) {
            throw new InvalidOperationException('Not a tensor');
        }

        if (DataType::STRING === $this->dataType) {
            throw new InvalidOperationException('Cannot get raw data for string tensor');
        }

        return Lib::api()->getTensorMutableData($this->handle);
    }

    /**
     * Get number of elements in sequence.
     *
     * @throws InvalidOperationException If not a sequence
     */
    public function sequenceLength(): int
    {
        if (!$this->type->isSequence()) {
            throw new InvalidOperationException('Not a sequence');
        }

        return Lib::api()->getValueCount($this->handle);
    }

    /**
     * Get element at index in sequence.
     *
     * @param int $index Element index
     *
     * @throws InvalidOperationException If not a sequence
     */
    public function getSequenceElement(int $index): self
    {
        if (!$this->type->isSequence()) {
            throw new InvalidOperationException('Not a sequence');
        }

        $api = Lib::api();
        $allocator = $api->getAllocatorWithDefaultOptions();
        $handle = $api->getValue($this->handle, $index, $allocator);

        return self::fromHandle($handle);
    }

    /**
     * Iterate over sequence elements.
     *
     * @param callable<self, int> $callback Function(OrtValue $value, int $index)
     *
     * @throws InvalidOperationException If not a sequence
     */
    public function foreachSequenceElement(callable $callback): void
    {
        $count = $this->sequenceLength();
        for ($i = 0; $i < $count; ++$i) {
            $callback($this->getSequenceElement($i), $i);
        }
    }

    /**
     * Get map keys as OrtValue.
     *
     * @return self Keys tensor
     *
     * @throws InvalidOperationException If not a map
     */
    public function mapKeys(): self
    {
        if (!$this->type->isMap()) {
            throw new InvalidOperationException('Not a map');
        }

        $api = Lib::api();
        $allocator = $api->getAllocatorWithDefaultOptions();
        $handle = $api->getValue($this->handle, 0, $allocator);

        return self::fromHandle($handle);
    }

    /**
     * Get map values as OrtValue.
     *
     * @return self Values tensor or sequence
     *
     * @throws InvalidOperationException If not a map
     */
    public function mapValues(): self
    {
        if (!$this->type->isMap()) {
            throw new InvalidOperationException('Not a map');
        }

        $api = Lib::api();
        $allocator = $api->getAllocatorWithDefaultOptions();
        $handle = $api->getValue($this->handle, 1, $allocator);

        return self::fromHandle($handle);
    }

    private static function inferShape(array $data): array
    {
        $shape = [];
        $current = $data;

        while (\is_array($current)) {
            $shape[] = \count($current);
            if (empty($current)) {
                break;
            }
            $current = $current[0];
        }

        return $shape;
    }

    private static function flattenArray(array $data, array $shape): array
    {
        if (empty($data)) {
            return [];
        }

        $flat = [];
        self::flattenRecursive($data, $shape, $flat);

        return $flat;
    }

    private static function flattenRecursive(array $data, array $shape, array &$flat): void
    {
        if (empty($shape)) {
            $flat[] = $data;

            return;
        }

        $dim = array_shift($shape);
        for ($i = 0; $i < $dim; ++$i) {
            if (empty($shape)) {
                $flat[] = $data[$i];
            } else {
                self::flattenRecursive($data[$i], $shape, $flat);
            }
        }
    }

    private static function createStringTensorFromArray(array $data, array $shape): self
    {
        $api = Lib::api();
        $allocator = $api->getAllocatorWithDefaultOptions();

        $handle = $api->createTensorAsOrtValue($allocator, $shape, DataType::STRING->value);

        $flatStrings = self::flattenArray($data, $shape);

        $api->fillStringTensor($handle, $flatStrings);

        return new self($handle, OnnxType::TENSOR, DataType::STRING, $shape);
    }

    private function tensorToArray(): array
    {
        if (DataType::UNDEFINED === $this->dataType || null === $this->shape) {
            throw new InvalidOperationException('Tensor info not available');
        }

        if (DataType::STRING === $this->dataType) {
            return $this->stringTensorToArray($this->shape);
        }

        $data = $this->tensorRawData();
        $elementCount = $this->elementCount();

        $ffi = Lib::get();
        $ffiType = $this->dataType->ffiType();
        $typedData = $ffi->cast("{$ffiType}[{$elementCount}]", $data);

        return self::buildNestedArray($typedData, $this->shape, $this->dataType);
    }

    private function stringTensorToArray(array $shape): array
    {
        $api = Lib::api();
        $elementCount = $this->elementCount();

        $strings = $api->getStringTensorContent($this->handle, $elementCount);

        return self::buildNestedArray($strings, $shape, null);
    }

    private static function buildNestedArray($flatData, array $shape, ?DataType $dataType): array
    {
        if (empty($shape)) {
            return [];
        }

        $result = [];
        $index = 0;
        self::buildNestedRecursive($flatData, $shape, $result, $index, $dataType);

        return $result;
    }

    private static function buildNestedRecursive($flatData, array $shape, array &$result, int &$index, ?DataType $dataType): void
    {
        $dim = array_shift($shape);

        if (empty($shape)) {
            for ($j = 0; $j < $dim; ++$j) {
                $value = $flatData[$index++];
                $result[] = DataType::BOOL === $dataType ? (bool) $value : $value;
            }
        } else {
            for ($j = 0; $j < $dim; ++$j) {
                $subArray = [];
                self::buildNestedRecursive($flatData, $shape, $subArray, $index, $dataType);
                $result[] = $subArray;
            }
        }
    }

    private function sequenceToArray(): array
    {
        $result = [];
        $this->foreachSequenceElement(static function ($value) use (&$result) {
            $result[] = $value->toArray();
        });

        return $result;
    }

    private function mapToArray(): array
    {
        $keys = $this->mapKeys()->toArray();
        $values = $this->mapValues()->toArray();

        if (\is_array($keys[0] ?? null)) {
            $keys = array_merge(...$keys);
        }

        return array_combine($keys, $values);
    }
}

<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Metadata;

use PhpMlKit\ONNXRuntime\Enums\DataType;
use PhpMlKit\ONNXRuntime\Enums\OnnxType;

/**
 * Metadata for tensor nodes.
 */
class TensorMetadata extends NodeMetadata
{
    /**
     * @param array<int>    $shape         Shape dimensions (may contain -1 for dynamic)
     * @param array<string> $symbolicShape Symbolic dimension names (if available)
     */
    public function __construct(
        public readonly DataType $dataType,
        public readonly array $shape,
        public readonly array $symbolicShape = []
    ) {
        parent::__construct(OnnxType::TENSOR);
    }

    /**
     * Get the data type of the tensor.
     */
    public function getDataType(): DataType
    {
        return $this->dataType;
    }

    /**
     * Get the shape of the tensor.
     *
     * @return array<int> Shape dimensions
     */
    public function getShape(): array
    {
        return $this->shape;
    }

    /**
     * Get symbolic dimension names if available.
     *
     * @return array<string>
     */
    public function getSymbolicShape(): array
    {
        return $this->symbolicShape;
    }

    /**
     * Check if this tensor has dynamic dimensions.
     */
    public function hasDynamicDimensions(): bool
    {
        return \in_array(-1, $this->shape, true);
    }

    /**
     * Get the number of dimensions.
     */
    public function getRank(): int
    {
        return \count($this->shape);
    }

    /**
     * Check if this is a scalar (0-dimensional tensor).
     */
    public function isScalar(): bool
    {
        return 0 === $this->getRank();
    }

    /**
     * Convert to array representation.
     *
     * @return array<string, mixed>
     */
    public function toArray(): array
    {
        return [
            'type' => $this->type->name,
            'dataType' => $this->dataType->name,
            'shape' => $this->shape,
            'symbolicShape' => $this->symbolicShape,
        ];
    }
}

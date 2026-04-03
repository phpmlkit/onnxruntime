<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Metadata;

use PhpMlKit\ONNXRuntime\Enums\OnnxType;

/**
 * Abstract base class for node metadata.
 */
abstract class NodeMetadata
{
    public function __construct(
        public readonly OnnxType $type
    ) {}

    /**
     * Get the ONNX type of this node.
     */
    public function getType(): OnnxType
    {
        return $this->type;
    }

    /**
     * Check if this node is a tensor.
     */
    public function isTensor(): bool
    {
        return $this->type->isTensor();
    }

    /**
     * Check if this node is a sequence.
     */
    public function isSequence(): bool
    {
        return $this->type->isSequence();
    }

    /**
     * Check if this node is a map.
     */
    public function isMap(): bool
    {
        return $this->type->isMap();
    }

    /**
     * Check if this node is optional.
     */
    public function isOptional(): bool
    {
        return $this->type->isOptional();
    }

    /**
     * Convert to array representation.
     *
     * @return array<string, mixed>
     */
    abstract public function toArray(): array;
}

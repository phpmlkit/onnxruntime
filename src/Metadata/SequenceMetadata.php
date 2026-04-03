<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Metadata;

use PhpMlKit\ONNXRuntime\Enums\OnnxType;

/**
 * Metadata for sequence nodes.
 */
class SequenceMetadata extends NodeMetadata
{
    public function __construct(
        public readonly MapMetadata|self|TensorMetadata $elementMetadata
    ) {
        parent::__construct(OnnxType::SEQUENCE);
    }

    /**
     * Get metadata for elements in the sequence.
     */
    public function getElementMetadata(): MapMetadata|self|TensorMetadata
    {
        return $this->elementMetadata;
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
            'elementMetadata' => $this->elementMetadata->toArray(),
        ];
    }
}

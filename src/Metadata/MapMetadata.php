<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Metadata;

use PhpMlKit\ONNXRuntime\Enums\DataType;
use PhpMlKit\ONNXRuntime\Enums\OnnxType;

/**
 * Metadata for map nodes.
 */
class MapMetadata extends NodeMetadata
{
    public function __construct(
        public readonly DataType $keyType,
        public readonly NodeMetadata $valueMetadata
    ) {
        parent::__construct(OnnxType::MAP);
    }

    /**
     * Get the key type of the map.
     */
    public function getKeyType(): DataType
    {
        return $this->keyType;
    }

    /**
     * Get metadata for values in the map.
     */
    public function getValueMetadata(): NodeMetadata
    {
        return $this->valueMetadata;
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
            'keyType' => $this->keyType->name,
            'valueMetadata' => $this->valueMetadata->toArray(),
        ];
    }
}

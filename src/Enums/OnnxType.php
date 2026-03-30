<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Enums;

/**
 * ONNX value types supported by ONNX Runtime.
 *
 * Mirrors the ONNXType enum from onnxruntime_c_api.h
 *
 * @see https://onnxruntime.ai/docs/api/c/struct_ort_api.html
 */
enum OnnxType: int
{
    case UNKNOWN = 0;
    case TENSOR = 1;
    case SEQUENCE = 2;
    case MAP = 3;
    case OPAQUE = 4;
    case SPARSE_TENSOR = 5;
    case OPTIONAL = 6;

    /**
     * Check if this type is a tensor (dense or sparse).
     */
    public function isTensor(): bool
    {
        return self::TENSOR === $this || self::SPARSE_TENSOR === $this;
    }

    /**
     * Check if this type is a composite type (sequence, map, or optional).
     */
    public function isComposite(): bool
    {
        return self::SEQUENCE === $this || self::MAP === $this || self::OPTIONAL === $this;
    }

    /**
     * Check if this type is a map.
     */
    public function isMap(): bool
    {
        return self::MAP === $this;
    }

    /**
     * Check if this type is a sequence.
     */
    public function isSequence(): bool
    {
        return self::SEQUENCE === $this;
    }

    /**
     * Check if this type is optional.
     */
    public function isOptional(): bool
    {
        return self::OPTIONAL === $this;
    }

    /**
     * Get human-readable name.
     */
    public function getName(): string
    {
        return match ($this) {
            self::UNKNOWN => 'unknown',
            self::TENSOR => 'tensor',
            self::SEQUENCE => 'sequence',
            self::MAP => 'map',
            self::OPAQUE => 'opaque',
            self::SPARSE_TENSOR => 'sparse_tensor',
            self::OPTIONAL => 'optional',
        };
    }
}

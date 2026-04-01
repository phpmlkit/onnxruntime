<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Enums;

/**
 * Type of API used to append an execution provider.
 */
enum ProviderApiType
{
    /**
     * Legacy API with simple device_id parameter.
     * Example: SessionOptionsAppendExecutionProvider_CUDA(options, device_id).
     */
    case LEGACY;

    /**
     * V2 API with complex options struct.
     * Requires Create/Update/Release lifecycle management.
     * Example: CreateCUDAProviderOptions -> UpdateCUDAProviderOptions -> Append -> Release.
     */
    case V2;

    /**
     * Generic API for providers that don't have dedicated functions.
     * Uses SessionOptionsAppendExecutionProvider with key-value options.
     */
    case GENERIC;
}

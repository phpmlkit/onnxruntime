<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Enums;

/**
 * CoreML MLComputeUnits configuration.
 *
 * Specifies which compute units CoreML should use for model execution.
 *
 * @see https://developer.apple.com/documentation/coreml/mlcomputeunits
 */
enum CoreMLComputeUnits: string
{
    /**
     * Use all available compute units (CPU, GPU, and Neural Engine).
     * Best performance but may use more power.
     */
    case ALL = 'All';

    /**
     * Use CPU and Neural Engine only.
     * Good balance of performance and power efficiency.
     */
    case CPU_AND_NEURAL_ENGINE = 'CPUAndNeuralEngine';

    /**
     * Use CPU and GPU only.
     * Good for older devices without Neural Engine.
     */
    case CPU_AND_GPU = 'CPUAndGPU';

    /**
     * Use CPU only.
     * Most compatible but slowest. Useful for validation and debugging.
     */
    case CPU_ONLY = 'CPUOnly';
}

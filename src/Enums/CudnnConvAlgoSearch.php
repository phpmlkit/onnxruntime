<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Enums;

/**
 * CUDNN convolution algorithm search options for CUDA provider.
 */
enum CudnnConvAlgoSearch: int
{
    /**
     * Exhaustive search for the best algorithm.
     * Most thorough but slowest initialization.
     */
    case EXHAUSTIVE = 0;

    /**
     * Heuristic-based algorithm selection.
     * Faster initialization than exhaustive.
     */
    case HEURISTIC = 1;

    /**
     * Default algorithm selection.
     * Fastest initialization, may not be optimal.
     */
    case DEFAULT = 2;
}

<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Enums;

/**
 * Graph optimization level to use with SessionOptions.
 *
 * @see https://www.onnxruntime.ai/docs/performance/graph-optimizations.html#graph-optimization-levels
 */
enum GraphOptimizationLevel: int
{
    case DISABLE_ALL = 0;
    case ENABLE_BASIC = 1;
    case ENABLE_EXTENDED = 2;
    case ENABLE_ALL = 99;
}

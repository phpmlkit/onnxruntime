<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Enums;

/**
 * DirectML performance preference options.
 */
enum DirectMLPerformancePreference: int
{
    case DEFAULT = 0;
    case HIGH_PERFORMANCE = 1;
    case MINIMUM_POWER = 2;
}

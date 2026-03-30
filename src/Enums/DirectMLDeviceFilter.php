<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Enums;

/**
 * DirectML device filter options (bitmask).
 */
enum DirectMLDeviceFilter: int
{
    case GPU = 1 << 0;
}

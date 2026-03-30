<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Enums;

/**
 * Memory types for allocated memory.
 */
enum MemoryType: int
{
    /** Any CPU memory used by non-CPU execution provider */
    case CPU_INPUT = -2;

    /** CPU accessible memory outputted by non-CPU execution provider, i.e. HOST_ACCESSIBLE */
    case CPU = -1;

    /** The default allocator for execution provider */
    case DEFAULT = 0;
}

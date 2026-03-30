<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Enums;

/**
 * Allocator types for allocated memory.
 */
enum AllocatorType: int
{
    case INVALID_ALLOCATOR = -1;
    case DEVICE_ALLOCATOR = 0;
    case ARENA_ALLOCATOR = 1;
    case READ_ONLY_ALLOCATOR = 2;
}

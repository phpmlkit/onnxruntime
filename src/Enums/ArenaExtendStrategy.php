<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Enums;

/**
 * Memory arena extension strategy for GPU execution providers.
 */
enum ArenaExtendStrategy: int
{
    /**
     * Extend arena to the next power of two.
     * More memory efficient, may have more allocations.
     */
    case NEXT_POWER_OF_TWO = 0;

    /**
     * Extend arena by the exact requested size.
     * Less memory waste, more frequent allocations.
     */
    case SAME_AS_REQUESTED = 1;
}

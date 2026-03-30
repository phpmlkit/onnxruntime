<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Enums;

/**
 * Logging severity levels.
 *
 * In typical API usage, specifying a logging severity level specifies the minimum severity of log messages to show.
 */
enum LoggingLevel: int
{
    /** Verbose informational messages (least severe). */
    case VERBOSE = 0;

    /** Informational messages. */
    case INFO = 1;

    /** Warning messages. */
    case WARNING = 2;

    /** Error messages. */
    case ERROR = 3;

    /** Fatal error messages (most severe). */
    case FATAL = 4;
}

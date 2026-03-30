<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Exceptions;

use PhpMlKit\ONNXRuntime\Exception;

/**
 * Generic failure exception.
 *
 * Thrown when an operation fails without a more specific error code.
 */
class FailException extends Exception {}

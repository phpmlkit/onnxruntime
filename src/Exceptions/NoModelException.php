<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Exceptions;

use PhpMlKit\ONNXRuntime\Exception;

/**
 * No model exception.
 *
 * Thrown when an operation requires a model but none is loaded.
 */
class NoModelException extends Exception {}

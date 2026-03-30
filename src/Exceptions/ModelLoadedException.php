<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Exceptions;

use PhpMlKit\ONNXRuntime\Exception;

/**
 * Model loaded exception.
 *
 * Thrown when an operation conflicts with an already loaded model.
 */
class ModelLoadedException extends Exception {}

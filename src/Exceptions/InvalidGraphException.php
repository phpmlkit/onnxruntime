<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Exceptions;

use PhpMlKit\ONNXRuntime\Exception;

/**
 * Invalid graph exception.
 *
 * Thrown when a model has an invalid computational graph.
 */
class InvalidGraphException extends Exception {}

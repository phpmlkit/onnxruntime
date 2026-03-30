<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Exceptions;

use PhpMlKit\ONNXRuntime\Exception;

/**
 * Invalid protobuf exception.
 *
 * Thrown when a model file has an invalid protobuf format.
 */
class InvalidProtobufException extends Exception {}

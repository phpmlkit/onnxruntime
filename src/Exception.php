<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime;

/**
 * Base exception for all ONNX Runtime errors.
 *
 * All ONNX Runtime-specific exceptions extend this class.
 */
abstract class Exception extends \Exception
{
    public function __construct(string $message = '', int $code = 0, ?\Throwable $previous = null)
    {
        parent::__construct($message, $code, $previous);
    }
}

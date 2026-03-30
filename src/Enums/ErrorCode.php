<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Enums;

use PhpMlKit\ONNXRuntime\Exception;
use PhpMlKit\ONNXRuntime\Exceptions\EngineErrorException;
use PhpMlKit\ONNXRuntime\Exceptions\ExecutionProviderException;
use PhpMlKit\ONNXRuntime\Exceptions\FailException;
use PhpMlKit\ONNXRuntime\Exceptions\InvalidArgumentException;
use PhpMlKit\ONNXRuntime\Exceptions\InvalidGraphException;
use PhpMlKit\ONNXRuntime\Exceptions\InvalidProtobufException;
use PhpMlKit\ONNXRuntime\Exceptions\ModelLoadedException;
use PhpMlKit\ONNXRuntime\Exceptions\NoModelException;
use PhpMlKit\ONNXRuntime\Exceptions\NoSuchFileException;
use PhpMlKit\ONNXRuntime\Exceptions\NotImplementedException;
use PhpMlKit\ONNXRuntime\Exceptions\RuntimeException;

/**
 * ONNX Runtime error codes.
 */
enum ErrorCode: int
{
    case OK = 0;                           // Success
    case FAIL = 1;                         // Generic failure
    case INVALID_ARGUMENT = 2;             // Invalid argument
    case NO_SUCH_FILE = 3;                 // File not found
    case NO_MODEL = 4;                     // No model loaded
    case ENGINE_ERROR = 5;                 // Engine error
    case RUNTIME_EXCEPTION = 6;            // Runtime exception
    case INVALID_PROTOBUF = 7;             // Invalid protobuf
    case MODEL_LOADED = 8;                 // Model already loaded
    case NOT_IMPLEMENTED = 9;              // Not implemented
    case INVALID_GRAPH = 10;               // Invalid graph
    case EP_FAIL = 11;                     // Execution provider failed

    /**
     * Get the exception class name for this error code.
     *
     * @return class-string<Exception>
     */
    public function getExceptionClass(): string
    {
        return match ($this) {
            self::OK => Exception::class,
            self::FAIL => FailException::class,
            self::INVALID_ARGUMENT => InvalidArgumentException::class,
            self::NO_SUCH_FILE => NoSuchFileException::class,
            self::NO_MODEL => NoModelException::class,
            self::ENGINE_ERROR => EngineErrorException::class,
            self::RUNTIME_EXCEPTION => RuntimeException::class,
            self::INVALID_PROTOBUF => InvalidProtobufException::class,
            self::MODEL_LOADED => ModelLoadedException::class,
            self::NOT_IMPLEMENTED => NotImplementedException::class,
            self::INVALID_GRAPH => InvalidGraphException::class,
            self::EP_FAIL => ExecutionProviderException::class,
        };
    }
}

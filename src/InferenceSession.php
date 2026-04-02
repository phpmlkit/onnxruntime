<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime;

use FFI\CData;
use PhpMlKit\ONNXRuntime\Contracts\Disposable;
use PhpMlKit\ONNXRuntime\Enums\AllocatorType;
use PhpMlKit\ONNXRuntime\Enums\DataType;
use PhpMlKit\ONNXRuntime\Enums\LoggingLevel;
use PhpMlKit\ONNXRuntime\Enums\MemoryType;
use PhpMlKit\ONNXRuntime\Exceptions\InvalidArgumentException;
use PhpMlKit\ONNXRuntime\Exceptions\InvalidOperationException;
use PhpMlKit\ONNXRuntime\FFI\Lib;

/**
 * ONNX Runtime Inference Session.
 *
 * Provides a clean API for running ONNX models.
 * Works with OrtValue objects as the primary data interface.
 *
 * @example
 * $session = InferenceSession::fromFile('model.onnx');
 *
 * // Create input from PHP array
 * $input = OrtValue::fromArray([1.0, 2.0, 3.0], DataType::FLOAT, [3]);
 *
 * // Run inference
 * $outputs = $session->run(['input' => $input]);
 *
 * // Convert output to PHP array
 * $result = $outputs['output']->toArray();
 */
class InferenceSession implements Disposable
{
    private static ?CData $env = null;
    private static int $envRefCount = 0;

    private CData $session;
    private CData $memoryInfo;
    private array $inputMetadata = [];
    private array $outputMetadata = [];
    private bool $disposed = false;

    /**
     * Private constructor. Use factory methods.
     *
     * @param CData $session ONNX Runtime session handle
     */
    private function __construct(CData $session)
    {
        $api = Lib::api();
        $this->session = $session;
        $this->memoryInfo = $api->createCpuMemoryInfo(AllocatorType::ARENA_ALLOCATOR, MemoryType::DEFAULT);

        $this->cacheMetadata();
    }

    public function __destruct()
    {
        $this->dispose();
    }

    /**
     * Create an inference session from a model file.
     *
     * @param string              $path    Path to the ONNX model file
     * @param null|SessionOptions $options Session configuration options
     *
     * @throws Exceptions\NoSuchFileException      If file doesn't exist
     * @throws Exceptions\InvalidProtobufException If file is invalid
     */
    public static function fromFile(string $path, ?SessionOptions $options = null): self
    {
        $api = Lib::api();

        $env = self::env();
        ++self::$envRefCount;

        $options ??= SessionOptions::default();
        $session = $api->createSession($env, $path, $options);

        return new self($session);
    }

    /**
     * Create an inference session from model bytes.
     *
     * @param string              $bytes   Raw ONNX model bytes
     * @param null|SessionOptions $options Session configuration options
     *
     * @throws Exceptions\InvalidProtobufException If bytes are invalid
     */
    public static function fromBytes(string $bytes, ?SessionOptions $options = null): self
    {
        $api = Lib::api();

        $env = self::env();
        ++self::$envRefCount;

        $options ??= SessionOptions::default();
        $session = $api->createSessionFromArray($env, $bytes, $options);

        return new self($session);
    }

    /**
     * Run inference.
     *
     * @param array<string, OrtValue> $inputs      Named input OrtValues
     * @param null|array<string>      $outputNames Specific outputs to return (null for all)
     * @param null|RunOptions         $options     Run options (null for defaults)
     *
     * @return array<string, OrtValue> Named output OrtValues
     *
     * @throws InvalidArgumentException If inputs are invalid
     */
    public function run(array $inputs, ?array $outputNames = null, ?RunOptions $options = null): array
    {
        $this->ensureOpen();

        $api = Lib::api();

        $inputNames = array_keys($this->inputMetadata);
        $outputNames ??= array_keys($this->outputMetadata);

        $inputValues = $this->prepareInputs($inputs, $inputNames);
        $options ??= RunOptions::default();

        $outputValues = $api->run($this->session, $options, $inputNames, $inputValues, $outputNames);

        $result = [];
        foreach ($outputNames as $i => $name) {
            $result[$name] = $outputValues[$i];
        }

        return $result;
    }

    /**
     * Get input metadata.
     *
     * @return array<string, array{name: string, shape: int[], dtype: DataType}>
     */
    public function inputs(): array
    {
        return $this->inputMetadata;
    }

    /**
     * Get output metadata.
     *
     * @return array<string, array{name: string, shape: int[], dtype: DataType}>
     */
    public function outputs(): array
    {
        return $this->outputMetadata;
    }

    /**
     * Close the session and release resources.
     */
    public function dispose(): void
    {
        if ($this->disposed) {
            return;
        }

        $api = Lib::api();

        if (isset($this->session)) {
            $api->releaseSession($this->session);
        }

        if (isset($this->memoryInfo)) {
            $api->releaseMemoryInfo($this->memoryInfo);
        }

        --self::$envRefCount;
        if (0 === self::$envRefCount && null !== self::$env) {
            $api->releaseEnv(self::$env);
            self::$env = null;
        }

        $this->disposed = true;
    }

    /**
     * Prepare inputs for inference.
     *
     * @param array<string, OrtValue> $inputs             Named input OrtValues
     * @param array<string>           $expectedInputNames Expected input names
     *
     * @return array<OrtValue> OrtValues in expected order
     *
     * @throws InvalidArgumentException If inputs are missing or unexpected
     */
    private function prepareInputs(array $inputs, array $expectedInputNames): array
    {
        $providedInputNames = [];
        $ortValues = [];

        foreach ($inputs as $name => $ortValue) {
            if (!$ortValue instanceof OrtValue) {
                throw new InvalidArgumentException(
                    "Input '{$name}' must be an OrtValue instance"
                );
            }

            $providedInputNames[] = $name;
            $ortValues[$name] = $ortValue;
        }

        $missing = array_diff($expectedInputNames, $providedInputNames);
        if (!empty($missing)) {
            throw new InvalidArgumentException(
                'Missing required input(s): '.implode(', ', $missing)
            );
        }

        $unexpected = array_diff($providedInputNames, $expectedInputNames);
        if (!empty($unexpected)) {
            throw new InvalidArgumentException(
                'Unexpected input(s): '.implode(', ', $unexpected)
                    .'. Expected: '.implode(', ', $expectedInputNames)
            );
        }

        // Return in expected order
        $ordered = [];
        foreach ($expectedInputNames as $name) {
            if (isset($ortValues[$name])) {
                $ordered[] = $ortValues[$name];
            }
        }

        return $ordered;
    }

    /**
     * Cache input and output metadata.
     */
    private function cacheMetadata(): void
    {
        $api = Lib::api();

        $inputCount = $api->sessionGetInputCount($this->session);

        for ($i = 0; $i < $inputCount; ++$i) {
            $typeInfo = $api->sessionGetInputTypeInfo($this->session, $i);

            try {
                $name = $this->getInputName($i);
                $this->inputMetadata[$name] = [
                    'name' => $name,
                    'shape' => $this->getInputShape($typeInfo),
                    'dtype' => $this->getInputType($typeInfo),
                ];
            } finally {
                $api->releaseTypeInfo($typeInfo);
            }
        }

        $outputCount = $api->sessionGetOutputCount($this->session);

        for ($i = 0; $i < $outputCount; ++$i) {
            $typeInfo = $api->sessionGetOutputTypeInfo($this->session, $i);

            try {
                $name = $this->getOutputName($i);
                $this->outputMetadata[$name] = [
                    'name' => $name,
                    'shape' => $this->getOutputShape($typeInfo),
                    'dtype' => $this->getOutputType($typeInfo),
                ];
            } finally {
                $api->releaseTypeInfo($typeInfo);
            }
        }
    }

    /**
     * Get input name.
     *
     * @param int $index Input index
     *
     * @return string Input name
     */
    private function getInputName(int $index): string
    {
        $api = Lib::api();
        $allocator = $api->getAllocatorWithDefaultOptions();

        return $api->sessionGetInputName($this->session, $index, $allocator);
    }

    /**
     * Get input shape.
     *
     * @param CData $typeInfo Input type info
     *
     * @return int[] Input shape
     */
    private function getInputShape(CData $typeInfo): array
    {
        return $this->getShapeFromTypeInfo($typeInfo);
    }

    /**
     * Get input type.
     *
     * @param CData $typeInfo Input type info
     *
     * @return DataType Input element type
     */
    private function getInputType(CData $typeInfo): DataType
    {
        return $this->getElementTypeFromTypeInfo($typeInfo);
    }

    /**
     * Get output name.
     *
     * @param int $index Output index
     *
     * @return string Output name
     */
    private function getOutputName(int $index): string
    {
        $api = Lib::api();
        $allocator = $api->getAllocatorWithDefaultOptions();

        return $api->sessionGetOutputName($this->session, $index, $allocator);
    }

    /**
     * Get output shape.
     *
     * @param CData $typeInfo Output type info
     *
     * @return int[] Output shape
     */
    private function getOutputShape(CData $typeInfo): array
    {
        return $this->getShapeFromTypeInfo($typeInfo);
    }

    /**
     * Get output element type.
     *
     * @param CData $typeInfo Output type info
     *
     * @return DataType Output element type
     */
    private function getOutputType(CData $typeInfo): DataType
    {
        return $this->getElementTypeFromTypeInfo($typeInfo);
    }

    /**
     * Get shape from type info.
     *
     * @param CData $typeInfo OrtTypeInfo pointer
     *
     * @return int[] Shape
     */
    private function getShapeFromTypeInfo(CData $typeInfo): array
    {
        $api = Lib::api();
        $tensorInfo = $api->castTypeInfoToTensorInfo($typeInfo);

        return $api->getDimensions($tensorInfo);
    }

    /**
     * Get element type from type info.
     *
     * @param CData $typeInfo OrtTypeInfo pointer
     *
     * @return DataType Element type
     */
    private function getElementTypeFromTypeInfo(CData $typeInfo): DataType
    {
        $api = Lib::api();
        $tensorInfo = $api->castTypeInfoToTensorInfo($typeInfo);

        return $api->getTensorElementType($tensorInfo);
    }

    /**
     * Ensure session is open.
     *
     * @throws InvalidOperationException If session is disposed
     */
    private function ensureOpen(): void
    {
        if ($this->disposed) {
            throw new InvalidOperationException('Inference session has been disposed');
        }
    }

    /**
     * Get environment.
     *
     * @return CData Environment
     */
    private static function env(): CData
    {
        if (null === self::$env) {
            $api = Lib::api();
            self::$env = $api->createEnv(LoggingLevel::FATAL, 'onnxruntime');

            // https://github.com/microsoft/onnxruntime/blob/main/docs/Privacy.md
            $api->disableTelemetryEvents(self::$env);
        }

        return self::$env;
    }
}

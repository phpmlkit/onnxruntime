<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime;

use FFI\CData;
use PhpMlKit\NDArray\NDArray;
use PhpMlKit\ONNXRuntime\Contracts\Disposable;
use PhpMlKit\ONNXRuntime\Enums\AllocatorType;
use PhpMlKit\ONNXRuntime\Enums\DataType;
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
    private CData $handle;
    private CData $memoryInfo;
    private array $inputMetadata = [];
    private array $outputMetadata = [];
    private bool $disposed = false;
    private Environment $environment;

    /**
     * Private constructor. Use factory methods.
     *
     * @param Environment $environment The environment managing this session
     */
    private function __construct(CData $handle, Environment $environment)
    {
        $this->handle = $handle;
        $this->environment = $environment;
        $api = Lib::api();
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

        $environment = Environment::acquire();

        $options ??= SessionOptions::default();
        $handle = $api->createSession($environment->getHandle(), $path, $options);

        return new self($handle, $environment);
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

        $environment = Environment::acquire();

        $options ??= SessionOptions::default();
        $handle = $api->createSessionFromArray($environment->getHandle(), $bytes, $options);

        return new self($handle, $environment);
    }

    /**
     * Run inference.
     *
     * Supports both OrtValue and NDArray inputs. All inputs must be of the same type.
     * Returns outputs in the same type as the inputs.
     *
     * @template T of OrtValue|\PhpMlKit\NDArray\NDArray
     *
     * @param array<string, T>   $inputs      Named input values (OrtValue or NDArray)
     * @param null|array<string> $outputNames Specific outputs to return (null for all)
     * @param null|RunOptions    $options     Run options (null for defaults)
     *
     * @return array<string, T> Named output values (same type as inputs)
     *
     * @throws InvalidArgumentException If inputs are invalid or mixed types
     */
    public function run(array $inputs, ?array $outputNames = null, ?RunOptions $options = null): array
    {
        $this->ensureOpen();

        $inputNames = array_keys($this->inputMetadata);
        $outputNames ??= array_keys($this->outputMetadata);

        $inputType = $this->validateInputs($inputs, $inputNames);
        $inputValues = $this->prepareInputs($inputs, $inputNames, $inputType);

        $api = Lib::api();
        $options ??= RunOptions::default();
        $outputOrtValues = $api->run($this->handle, $options, $inputNames, $inputValues, $outputNames);

        return $this->prepareOutputs($outputOrtValues, $outputNames, $inputType);
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
     * Dispose the session and release resources.
     *
     * Releases the session and memory info, and decrements the environment
     * reference count. When the last session using the environment is disposed,
     * the environment itself is freed.
     */
    public function dispose(): void
    {
        if ($this->disposed) {
            return;
        }

        $api = Lib::api();

        $api->releaseSession($this->handle);

        $api->releaseMemoryInfo($this->memoryInfo);

        $this->environment->dispose();

        $this->disposed = true;
    }

    /**
     * Validate inputs and detect their type.
     *
     * Checks that:
     * - All inputs are of the same type (OrtValue or NDArray)
     * - No required inputs are missing
     * - No unexpected inputs are provided
     *
     * @param array<string, mixed> $inputs             Named input values
     * @param array<string>        $expectedInputNames Expected input names from model
     *
     * @return string 'ortvalue' or 'ndarray'
     *
     * @throws InvalidArgumentException If validation fails
     */
    private function validateInputs(array $inputs, array $expectedInputNames): string
    {
        if (empty($inputs)) {
            throw new InvalidArgumentException('Inputs cannot be empty');
        }

        $inputType = null;
        $ndarrayClass = NDArray::class;

        foreach ($inputs as $name => $value) {
            $isNDArray = class_exists($ndarrayClass) && $value instanceof $ndarrayClass;
            $isOrtValue = $value instanceof OrtValue;

            if (!$isNDArray && !$isOrtValue) {
                throw new InvalidArgumentException(
                    "Input '{$name}' must be an OrtValue or NDArray instance, "
                    .\gettype($value).' given'
                );
            }

            $actualType = $isNDArray ? 'ndarray' : 'ortvalue';
            $inputType ??= $actualType;

            if ($actualType !== $inputType) {
                throw new InvalidArgumentException(
                    "Mixed input types not supported. Input '{$name}' is {$actualType}, "
                    ."but expected all inputs to be {$inputType}"
                );
            }
        }

        \assert(null !== $inputType);

        $providedNames = array_keys($inputs);
        $missing = array_diff($expectedInputNames, $providedNames);
        if (!empty($missing)) {
            throw new InvalidArgumentException(
                'Missing required input(s): '.implode(', ', $missing)
            );
        }

        $unexpected = array_diff($providedNames, $expectedInputNames);
        if (!empty($unexpected)) {
            throw new InvalidArgumentException(
                'Unexpected input(s): '.implode(', ', $unexpected)
                .'. Expected: '.implode(', ', $expectedInputNames)
            );
        }

        return $inputType;
    }

    /**
     * Prepare inputs for inference.
     *
     * Converts NDArray inputs to OrtValue and orders them correctly.
     *
     * @param array<string, mixed> $inputs     Named input values
     * @param array<string>        $inputNames Input names in order
     * @param 'ndarray'|'ortvalue' $inputType  Input type
     *
     * @return array<OrtValue> OrtValues in expected order
     */
    private function prepareInputs(array $inputs, array $inputNames, string $inputType): array
    {
        $ordered = [];

        foreach ($inputNames as $name) {
            $value = $inputs[$name];
            $ordered[] = 'ndarray' === $inputType ? OrtValue::fromNDArray($value) : $value;
        }

        return $ordered;
    }

    /**
     * Prepare outputs after inference.
     *
     * Converts OrtValue outputs to NDArray if inputs were NDArray.
     *
     * @param array<OrtValue>      $outputOrtValues Output values from inference
     * @param array<string>        $outputNames     Output names in order
     * @param 'ndarray'|'ortvalue' $inputType       Input type
     *
     * @return array<string, NDArray|OrtValue> Named outputs
     */
    private function prepareOutputs(array $outputOrtValues, array $outputNames, string $inputType): array
    {
        $result = [];

        foreach ($outputNames as $i => $name) {
            $ortValue = $outputOrtValues[$i];
            $result[$name] = 'ndarray' === $inputType ? $ortValue->toNDArray() : $ortValue;
        }

        return $result;
    }

    /**
     * Cache input and output metadata.
     */
    private function cacheMetadata(): void
    {
        $api = Lib::api();

        $inputCount = $api->sessionGetInputCount($this->handle);

        for ($i = 0; $i < $inputCount; ++$i) {
            $typeInfo = $api->sessionGetInputTypeInfo($this->handle, $i);

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

        $outputCount = $api->sessionGetOutputCount($this->handle);

        for ($i = 0; $i < $outputCount; ++$i) {
            $typeInfo = $api->sessionGetOutputTypeInfo($this->handle, $i);

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

        return $api->sessionGetInputName($this->handle, $index, $allocator);
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

        return $api->sessionGetOutputName($this->handle, $index, $allocator);
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
}

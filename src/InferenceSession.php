<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime;

use FFI\CData;
use PhpMlKit\ONNXRuntime\Contracts\Disposable;
use PhpMlKit\ONNXRuntime\Enums\AllocatorType;
use PhpMlKit\ONNXRuntime\Enums\MemoryType;
use PhpMlKit\ONNXRuntime\Enums\OnnxType;
use PhpMlKit\ONNXRuntime\Exceptions\InvalidArgumentException;
use PhpMlKit\ONNXRuntime\Exceptions\InvalidOperationException;
use PhpMlKit\ONNXRuntime\Exceptions\NotImplementedException;
use PhpMlKit\ONNXRuntime\FFI\Lib;
use PhpMlKit\ONNXRuntime\Metadata\MapMetadata;
use PhpMlKit\ONNXRuntime\Metadata\ModelMetadata;
use PhpMlKit\ONNXRuntime\Metadata\SequenceMetadata;
use PhpMlKit\ONNXRuntime\Metadata\TensorMetadata;

/**
 * ONNX Runtime Inference Session.
 *
 * Provides a clean API for running ONNX models.
 * Works with OrtValue objects as the primary data interface.
 */
class InferenceSession implements Disposable
{
    private CData $memoryInfo;
    private CData $allocator;

    private ?ModelMetadata $modelMetadata = null;

    /** @var array<string, MapMetadata|SequenceMetadata|TensorMetadata> */
    private array $inputMetadata = [];

    /** @var array<string, MapMetadata|SequenceMetadata|TensorMetadata> */
    private array $outputMetadata = [];

    private bool $disposed = false;

    /**
     * Private constructor. Use factory methods.
     *
     * @param Environment $environment The environment managing this session
     */
    private function __construct(private readonly CData $handle, private readonly Environment $environment)
    {
        $api = Lib::api();

        $this->memoryInfo = $api->createCpuMemoryInfo(AllocatorType::ARENA_ALLOCATOR, MemoryType::DEFAULT);
        $this->allocator = $api->getAllocatorWithDefaultOptions();

        $this->cacheNodeMetadata();
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

        $environment = Environment::instance();
        $options ??= SessionOptions::default();

        $handle = $api->createSession($environment, $path, $options);

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

        $environment = Environment::instance();
        $options ??= SessionOptions::default();

        $handle = $api->createSessionFromArray($environment, $bytes, $options);

        return new self($handle, $environment);
    }

    /**
     * Run inference.
     *
     * @param array<string, OrtValue> $inputs      Named input values
     * @param null|array<string>      $outputNames Specific outputs to return (null for all)
     * @param null|RunOptions         $options     Run options (null for defaults)
     *
     * @return array<string, OrtValue> Named output values
     *
     * @throws InvalidArgumentException  If inputs are invalid
     * @throws InvalidOperationException If session is disposed
     */
    public function run(array $inputs, ?array $outputNames = null, ?RunOptions $options = null): array
    {
        $this->ensureOpen();

        $api = Lib::api();

        $inputNames = $this->inputNames();
        $outputNames ??= $this->outputNames();

        $this->validateInputs($inputs, $inputNames);

        $inputValues = array_map(static fn ($name) => $inputs[$name], $inputNames);
        $options ??= RunOptions::default();

        $outputValues = $api->run($this->handle, $options, $inputNames, $inputValues, $outputNames);

        return array_combine($outputNames, $outputValues);
    }

    /**
     * Get input metadata.
     *
     * @return array<string, MapMetadata|SequenceMetadata|TensorMetadata>
     */
    public function inputs(): array
    {
        return $this->inputMetadata;
    }

    /**
     * Get output metadata.
     *
     * @return array<string, MapMetadata|SequenceMetadata|TensorMetadata>
     */
    public function outputs(): array
    {
        return $this->outputMetadata;
    }

    /**
     * Get input names.
     *
     * @return array<string>
     */
    public function inputNames(): array
    {
        return array_keys($this->inputMetadata);
    }

    /**
     * Get output names.
     *
     * @return array<string>
     */
    public function outputNames(): array
    {
        return array_keys($this->outputMetadata);
    }

    /**
     * Get model metadata.
     *
     * Lazily loads model metadata on first call.
     */
    public function metadata(): ModelMetadata
    {
        if (null === $this->modelMetadata) {
            $this->modelMetadata = new ModelMetadata($this);
        }

        return $this->modelMetadata;
    }

    /**
     * Get the session handle for internal use.
     *
     * @internal
     */
    public function getHandle(): CData
    {
        return $this->handle;
    }

    /**
     * Get the allocator for internal use.
     *
     * @internal
     */
    public function getAllocator(): CData
    {
        return $this->allocator;
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
     * Validate inputs.
     *
     * Checks that:
     * - All inputs are OrtValue objects
     * - No required inputs are missing
     * - No unexpected inputs are provided
     *
     * @param array<string, OrtValue> $inputs             Named input values
     * @param array<string>           $expectedInputNames Expected input names from model
     *
     * @throws InvalidArgumentException If validation fails
     */
    private function validateInputs(array $inputs, array $expectedInputNames): void
    {
        if (empty($inputs)) {
            throw new InvalidArgumentException('Inputs cannot be empty');
        }

        foreach ($inputs as $name => $value) {
            if (!$value instanceof OrtValue) {
                throw new InvalidArgumentException(
                    "Input '{$name}' must be an OrtValue instance, "
                        .\gettype($value).' given'
                );
            }
        }

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
    }

    /**
     * Cache input and output metadata.
     */
    private function cacheNodeMetadata(): void
    {
        $api = Lib::api();

        $inputCount = $api->sessionGetInputCount($this->handle);

        for ($i = 0; $i < $inputCount; ++$i) {
            $typeInfo = $api->sessionGetInputTypeInfo($this->handle, $i);

            try {
                $name = $api->sessionGetInputName($this->handle, $i, $this->allocator);
                $this->inputMetadata[$name] = $this->createMetadataFromTypeInfo($typeInfo);
            } finally {
                $api->releaseTypeInfo($typeInfo);
            }
        }

        $outputCount = $api->sessionGetOutputCount($this->handle);

        for ($i = 0; $i < $outputCount; ++$i) {
            $typeInfo = $api->sessionGetOutputTypeInfo($this->handle, $i);

            try {
                $name = $api->sessionGetOutputName($this->handle, $i, $this->allocator);
                $this->outputMetadata[$name] = $this->createMetadataFromTypeInfo($typeInfo);
            } finally {
                $api->releaseTypeInfo($typeInfo);
            }
        }
    }

    /**
     * Create metadata from type info.
     *
     * @param CData $typeInfo OrtTypeInfo pointer
     *
     * @throws NotImplementedException for unsupported types
     */
    private function createMetadataFromTypeInfo(CData $typeInfo): MapMetadata|SequenceMetadata|TensorMetadata
    {
        $api = Lib::api();
        $onnxType = $api->getOnnxTypeFromTypeInfo($typeInfo);

        return match ($onnxType) {
            OnnxType::TENSOR, OnnxType::SPARSE_TENSOR => $this->createTensorMetadata($typeInfo),
            OnnxType::SEQUENCE => $this->createSequenceMetadata($typeInfo),
            OnnxType::MAP => $this->createMapMetadata($typeInfo),
            OnnxType::OPTIONAL => throw new NotImplementedException('OPTIONAL type is not supported'),
            default => throw new NotImplementedException("ONNX type '{$onnxType->name}' is not supported"),
        };
    }

    /**
     * Create tensor metadata from type info.
     *
     * @param CData $typeInfo OrtTypeInfo pointer
     */
    private function createTensorMetadata(CData $typeInfo): TensorMetadata
    {
        $api = Lib::api();
        $tensorInfo = $api->castTypeInfoToTensorInfo($typeInfo);

        $dataType = $api->getTensorElementType($tensorInfo);
        $shape = $api->getDimensions($tensorInfo);
        $symbolicShape = $api->getSymbolicDimensions($tensorInfo);

        return new TensorMetadata($dataType, $shape, $symbolicShape);
    }

    /**
     * Create sequence metadata from type info.
     *
     * @param CData $typeInfo OrtTypeInfo pointer
     */
    private function createSequenceMetadata(CData $typeInfo): SequenceMetadata
    {
        $api = Lib::api();
        $sequenceInfo = $api->castTypeInfoToSequenceTypeInfo($typeInfo);

        $elementTypeInfo = $api->getSequenceElementType($sequenceInfo);

        try {
            $elementMetadata = $this->createMetadataFromTypeInfo($elementTypeInfo);

            return new SequenceMetadata($elementMetadata);
        } finally {
            $api->releaseTypeInfo($elementTypeInfo);
        }
    }

    /**
     * Create map metadata from type info.
     *
     * @param CData $typeInfo OrtTypeInfo pointer
     */
    private function createMapMetadata(CData $typeInfo): MapMetadata
    {
        $api = Lib::api();
        $mapInfo = $api->castTypeInfoToMapTypeInfo($typeInfo);

        $keyType = $api->getMapKeyType($mapInfo);
        $valueTypeInfo = $api->getMapValueType($mapInfo);

        try {
            $valueMetadata = $this->createMetadataFromTypeInfo($valueTypeInfo);

            return new MapMetadata($keyType, $valueMetadata);
        } finally {
            $api->releaseTypeInfo($valueTypeInfo);
        }
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

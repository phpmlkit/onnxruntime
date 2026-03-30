<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\FFI;

use FFI\CData;
use PhpMlKit\ONNXRuntime\Enums\AllocatorType;
use PhpMlKit\ONNXRuntime\Enums\DataType;
use PhpMlKit\ONNXRuntime\Enums\LoggingLevel;
use PhpMlKit\ONNXRuntime\Enums\MemoryType;
use PhpMlKit\ONNXRuntime\Enums\OnnxType;
use PhpMlKit\ONNXRuntime\Exception;
use PhpMlKit\ONNXRuntime\OrtValue;
use PhpMlKit\ONNXRuntime\RunOptions;
use PhpMlKit\ONNXRuntime\SessionOptions;

/**
 * High-level typed wrapper for ONNX Runtime C API via FFI.
 */
class Api
{
    private \FFI $ffi;
    private CData $api;

    public function __construct(\FFI $ffi, CData $api)
    {
        $this->ffi = $ffi;
        $this->api = $api;
    }

    /**
     * Create an ONNX Runtime environment.
     *
     * @param LoggingLevel $logSeverityLevel Logging severity level
     * @param string       $logId            Log identifier
     *
     * @return CData OrtEnv pointer
     *
     * @throws Exception on error
     */
    public function createEnv(LoggingLevel $logSeverityLevel, string $logId): CData
    {
        $env = $this->ffi->new('OrtEnv*');
        $status = ($this->api->CreateEnv)($logSeverityLevel->value, $logId, \FFI::addr($env));
        Lib::checkStatus($status);

        return $env;
    }

    /**
     * Release an environment.
     *
     * @param CData $env OrtEnv pointer
     */
    public function releaseEnv(CData $env): void
    {
        ($this->api->ReleaseEnv)($env);
    }

    /**
     * Create session options.
     *
     * @return CData OrtSessionOptions pointer
     *
     * @throws Exception on error
     */
    public function createSessionOptions(): CData
    {
        $options = $this->ffi->new('OrtSessionOptions*');
        $status = ($this->api->CreateSessionOptions)(\FFI::addr($options));
        Lib::checkStatus($status);

        return $options;
    }

    /**
     * Release session options.
     *
     * @param CData $options OrtSessionOptions pointer
     */
    public function releaseSessionOptions(CData $options): void
    {
        ($this->api->ReleaseSessionOptions)($options);
    }

    /**
     * Set session graph optimization level.
     *
     * @param CData $options OrtSessionOptions pointer
     * @param int   $level   Optimization level (0=disable, 1=basic, 2=extended, 99=all)
     *
     * @throws Exception on error
     */
    public function setSessionGraphOptimizationLevel(CData $options, int $level): void
    {
        $status = ($this->api->SetSessionGraphOptimizationLevel)($options, $level);
        Lib::checkStatus($status);
    }

    /**
     * Set session execution mode.
     *
     * @param CData $options OrtSessionOptions pointer
     * @param int   $mode    Execution mode (0=sequential, 1=parallel)
     *
     * @throws Exception on error
     */
    public function setSessionExecutionMode(CData $options, int $mode): void
    {
        $status = ($this->api->SetSessionExecutionMode)($options, $mode);
        Lib::checkStatus($status);
    }

    /**
     * Set number of inter-op threads.
     *
     * @param CData $options    OrtSessionOptions pointer
     * @param int   $numThreads Number of threads
     *
     * @throws Exception on error
     */
    public function setInterOpNumThreads(CData $options, int $numThreads): void
    {
        $status = ($this->api->SetInterOpNumThreads)($options, $numThreads);
        Lib::checkStatus($status);
    }

    /**
     * Set number of intra-op threads.
     *
     * @param CData $options    OrtSessionOptions pointer
     * @param int   $numThreads Number of threads
     *
     * @throws Exception on error
     */
    public function setIntraOpNumThreads(CData $options, int $numThreads): void
    {
        $status = ($this->api->SetIntraOpNumThreads)($options, $numThreads);
        Lib::checkStatus($status);
    }

    /**
     * Enable CPU memory arena.
     *
     * @param CData $options OrtSessionOptions pointer
     *
     * @throws Exception on error
     */
    public function enableCpuMemArena(CData $options): void
    {
        $status = ($this->api->EnableCpuMemArena)($options);
        Lib::checkStatus($status);
    }

    /**
     * Disable CPU memory arena.
     *
     * @param CData $options OrtSessionOptions pointer
     *
     * @throws Exception on error
     */
    public function disableCpuMemArena(CData $options): void
    {
        $status = ($this->api->DisableCpuMemArena)($options);
        Lib::checkStatus($status);
    }

    /**
     * Enable memory pattern optimization.
     *
     * @param CData $options OrtSessionOptions pointer
     *
     * @throws Exception on error
     */
    public function enableMemPattern(CData $options): void
    {
        $status = ($this->api->EnableMemPattern)($options);
        Lib::checkStatus($status);
    }

    /**
     * Disable memory pattern optimization.
     *
     * @param CData $options OrtSessionOptions pointer
     *
     * @throws Exception on error
     */
    public function disableMemPattern(CData $options): void
    {
        $status = ($this->api->DisableMemPattern)($options);
        Lib::checkStatus($status);
    }

    /**
     * Enable profiling.
     *
     * @param CData  $options           OrtSessionOptions pointer
     * @param string $profileFilePrefix Prefix for profile file names
     *
     * @throws Exception on error
     */
    public function enableProfiling(CData $options, string $profileFilePrefix): void
    {
        $status = ($this->api->EnableProfiling)($options, $profileFilePrefix);
        Lib::checkStatus($status);
    }

    /**
     * Disable profiling.
     *
     * @param CData $options OrtSessionOptions pointer
     *
     * @throws Exception on error
     */
    public function disableProfiling(CData $options): void
    {
        $status = ($this->api->DisableProfiling)($options);
        Lib::checkStatus($status);
    }

    /**
     * Set session log severity level.
     *
     * @param CData        $options OrtSessionOptions pointer
     * @param LoggingLevel $level   Log severity level
     *
     * @throws Exception on error
     */
    public function setSessionLogSeverityLevel(CData $options, LoggingLevel $level): void
    {
        $status = ($this->api->SetSessionLogSeverityLevel)($options, $level->value);
        Lib::checkStatus($status);
    }

    /**
     * Set session log verbosity level.
     *
     * @param CData        $options OrtSessionOptions pointer
     * @param LoggingLevel $level   Log verbosity level
     *
     * @throws Exception on error
     */
    public function setSessionLogVerbosityLevel(CData $options, LoggingLevel $level): void
    {
        $status = ($this->api->SetSessionLogVerbosityLevel)($options, $level->value);
        Lib::checkStatus($status);
    }

    /**
     * Create an inference session from a model file.
     *
     * @param CData          $env       OrtEnv pointer
     * @param string         $modelPath Path to the ONNX model file
     * @param SessionOptions $options   Session options
     *
     * @return CData OrtSession pointer
     *
     * @throws Exception on error
     */
    public function createSession(CData $env, string $modelPath, SessionOptions $options): CData
    {
        $session = $this->ffi->new('OrtSession*');
        $status = ($this->api->CreateSession)($env, $modelPath, $options->getHandle(), \FFI::addr($session));
        Lib::checkStatus($status);

        return $session;
    }

    /**
     * Create an inference session from model bytes.
     *
     * @param CData          $env        OrtEnv pointer
     * @param string         $modelBytes Raw ONNX model bytes
     * @param SessionOptions $options    Session options
     *
     * @return CData OrtSession pointer
     *
     * @throws Exception on error
     */
    public function createSessionFromArray(CData $env, string $modelBytes, SessionOptions $options): CData
    {
        $session = $this->ffi->new('OrtSession*');
        $status = ($this->api->CreateSessionFromArray)(
            $env,
            $modelBytes,
            \strlen($modelBytes),
            $options->getHandle(),
            \FFI::addr($session)
        );
        Lib::checkStatus($status);

        return $session;
    }

    /**
     * Release a session.
     *
     * @param CData $session OrtSession pointer
     */
    public function releaseSession(CData $session): void
    {
        ($this->api->ReleaseSession)($session);
    }

    /**
     * Get the number of model inputs.
     *
     * @param CData $session OrtSession pointer
     *
     * @return int Number of inputs
     *
     * @throws Exception on error
     */
    public function sessionGetInputCount(CData $session): int
    {
        $count = $this->ffi->new('size_t');
        $status = ($this->api->SessionGetInputCount)($session, \FFI::addr($count));
        Lib::checkStatus($status);

        return $count->cdata;
    }

    /**
     * Get the number of model outputs.
     *
     * @param CData $session OrtSession pointer
     *
     * @return int Number of outputs
     *
     * @throws Exception on error
     */
    public function sessionGetOutputCount(CData $session): int
    {
        $count = $this->ffi->new('size_t');
        $status = ($this->api->SessionGetOutputCount)($session, \FFI::addr($count));
        Lib::checkStatus($status);

        return $count->cdata;
    }

    /**
     * Get input name at specified index.
     *
     * @param CData $session   OrtSession pointer
     * @param int   $index     Input index
     * @param CData $allocator OrtAllocator pointer
     *
     * @return string Input name
     *
     * @throws Exception on error
     */
    public function sessionGetInputName(CData $session, int $index, CData $allocator): string
    {
        $name = $this->ffi->new('char*');
        $status = ($this->api->SessionGetInputName)($session, $index, $allocator, \FFI::addr($name));
        Lib::checkStatus($status);

        return \FFI::string($name);
    }

    /**
     * Get output name at specified index.
     *
     * @param CData $session   OrtSession pointer
     * @param int   $index     Output index
     * @param CData $allocator OrtAllocator pointer
     *
     * @return string Output name
     *
     * @throws Exception on error
     */
    public function sessionGetOutputName(CData $session, int $index, CData $allocator): string
    {
        $name = $this->ffi->new('char*');
        $status = ($this->api->SessionGetOutputName)($session, $index, $allocator, \FFI::addr($name));
        Lib::checkStatus($status);

        return \FFI::string($name);
    }

    /**
     * Get input type information.
     *
     * @param CData $session OrtSession pointer
     * @param int   $index   Input index
     *
     * @return CData OrtTypeInfo pointer
     *
     * @throws Exception on error
     */
    public function sessionGetInputTypeInfo(CData $session, int $index): CData
    {
        $typeInfo = $this->ffi->new('OrtTypeInfo*');
        $status = ($this->api->SessionGetInputTypeInfo)($session, $index, \FFI::addr($typeInfo));
        Lib::checkStatus($status);

        return $typeInfo;
    }

    /**
     * Get output type information.
     *
     * @param CData $session OrtSession pointer
     * @param int   $index   Output index
     *
     * @return CData OrtTypeInfo pointer
     *
     * @throws Exception on error
     */
    public function sessionGetOutputTypeInfo(CData $session, int $index): CData
    {
        $typeInfo = $this->ffi->new('OrtTypeInfo*');
        $status = ($this->api->SessionGetOutputTypeInfo)($session, $index, \FFI::addr($typeInfo));
        Lib::checkStatus($status);

        return $typeInfo;
    }

    /**
     * Create CPU memory information.
     *
     * @param AllocatorType $allocatorType Allocator type
     * @param MemoryType    $memType       Memory type
     *
     * @return CData OrtMemoryInfo pointer
     *
     * @throws Exception on error
     */
    public function createCpuMemoryInfo(AllocatorType $allocatorType, MemoryType $memType): CData
    {
        $memoryInfo = $this->ffi->new('OrtMemoryInfo*');
        $status = ($this->api->CreateCpuMemoryInfo)($allocatorType->value, $memType->value, \FFI::addr($memoryInfo));
        Lib::checkStatus($status);

        return $memoryInfo;
    }

    /**
     * Release memory information.
     *
     * @param CData $memoryInfo OrtMemoryInfo pointer
     */
    public function releaseMemoryInfo(CData $memoryInfo): void
    {
        ($this->api->ReleaseMemoryInfo)($memoryInfo);
    }

    /**
     * Create a tensor with data as an OrtValue.
     *
     * @param CData    $memoryInfo OrtMemoryInfo pointer
     * @param CData    $data       Data pointer
     * @param int      $dataLength Data length in bytes
     * @param array    $shape      Tensor shape
     * @param DataType $type       ONNX tensor element data type
     *
     * @return CData OrtValue pointer
     *
     * @throws Exception on error
     */
    public function createTensorWithDataAsOrtValue(
        CData $memoryInfo,
        CData $data,
        int $dataLength,
        array $shape,
        DataType $type
    ): CData {
        $shapeC = $this->ffi->new('int64_t['.\count($shape).']');
        foreach ($shape as $i => $dim) {
            $shapeC[$i] = $dim;
        }

        $value = $this->ffi->new('OrtValue*');
        $status = ($this->api->CreateTensorWithDataAsOrtValue)(
            $memoryInfo,
            $data,
            $dataLength,
            $shapeC,
            \count($shape),
            $type->value,
            \FFI::addr($value)
        );
        Lib::checkStatus($status);

        return $value;
    }

    /**
     * Get tensor type and shape information.
     *
     * @param CData $value OrtValue pointer
     *
     * @return CData OrtTensorTypeAndShapeInfo pointer
     *
     * @throws Exception on error
     */
    public function getTensorTypeAndShape(CData $value): CData
    {
        $tensorInfo = $this->ffi->new('OrtTensorTypeAndShapeInfo*');
        $status = ($this->api->GetTensorTypeAndShape)($value, \FFI::addr($tensorInfo));
        Lib::checkStatus($status);

        return $tensorInfo;
    }

    /**
     * Get tensor element type.
     *
     * @param CData $tensorInfo OrtTensorTypeAndShapeInfo pointer
     *
     * @return DataType Tensor element type
     *
     * @throws Exception on error
     */
    public function getTensorElementType(CData $tensorInfo): DataType
    {
        $dataType = $this->ffi->new('ONNXTensorElementDataType');
        $status = ($this->api->GetTensorElementType)($tensorInfo, \FFI::addr($dataType));
        Lib::checkStatus($status);

        return DataType::from($dataType->cdata);
    }

    /**
     * Get tensor dimensions count.
     *
     * @param CData $tensorInfo OrtTensorTypeAndShapeInfo pointer
     *
     * @return int Number of dimensions
     *
     * @throws Exception on error
     */
    public function getDimensionsCount(CData $tensorInfo): int
    {
        $count = $this->ffi->new('size_t');
        $status = ($this->api->GetDimensionsCount)($tensorInfo, \FFI::addr($count));
        Lib::checkStatus($status);

        return $count->cdata;
    }

    /**
     * Get tensor dimensions.
     *
     * @param CData $tensorInfo OrtTensorTypeAndShapeInfo pointer
     *
     * @return array Dimension values
     *
     * @throws Exception on error
     */
    public function getDimensions(CData $tensorInfo): array
    {
        $dimCount = $this->getDimensionsCount($tensorInfo);

        if (0 === $dimCount) {
            return [];
        }

        $dims = $this->ffi->new("int64_t[{$dimCount}]");
        $status = ($this->api->GetDimensions)($tensorInfo, $dims, $dimCount);
        Lib::checkStatus($status);

        $shape = [];
        for ($i = 0; $i < $dimCount; ++$i) {
            $shape[] = $dims[$i];
        }

        return $shape;
    }

    /**
     * Get mutable tensor data pointer.
     *
     * @param CData $value OrtValue pointer
     *
     * @return CData Data pointer
     *
     * @throws Exception on error
     */
    public function getTensorMutableData(CData $value): CData
    {
        $dataPtr = $this->ffi->new('void*');
        $status = ($this->api->GetTensorMutableData)($value, \FFI::addr($dataPtr));
        Lib::checkStatus($status);

        return $dataPtr;
    }

    /**
     * Release tensor type and shape info.
     *
     * @param CData $tensorInfo OrtTensorTypeAndShapeInfo pointer
     */
    public function releaseTensorTypeAndShapeInfo(CData $tensorInfo): void
    {
        ($this->api->ReleaseTensorTypeAndShapeInfo)($tensorInfo);
    }

    /**
     * Release an OrtValue.
     *
     * @param CData $value OrtValue pointer
     */
    public function releaseValue(CData $value): void
    {
        ($this->api->ReleaseValue)($value);
    }

    /**
     * Cast type info to tensor info.
     *
     * @param CData $typeInfo OrtTypeInfo pointer
     *
     * @return CData OrtTensorTypeAndShapeInfo pointer
     *
     * @throws Exception on error
     */
    public function castTypeInfoToTensorInfo(CData $typeInfo): CData
    {
        $tensorInfo = $this->ffi->new('OrtTensorTypeAndShapeInfo*');
        $status = ($this->api->CastTypeInfoToTensorInfo)($typeInfo, \FFI::addr($tensorInfo));
        Lib::checkStatus($status);

        return $tensorInfo;
    }

    /**
     * Get ONNX type from type info.
     *
     * @param CData $typeInfo OrtTypeInfo pointer
     *
     * @return int ONNX type enum value
     */
    public function getONNXTypeFromTypeInfo(CData $typeInfo): int
    {
        return ($this->api->GetONNXTypeFromTypeInfo)($typeInfo);
    }

    /**
     * Release type information.
     *
     * @param CData $typeInfo OrtTypeInfo pointer
     */
    public function releaseTypeInfo(CData $typeInfo): void
    {
        ($this->api->ReleaseTypeInfo)($typeInfo);
    }

    /**
     * Create run options.
     *
     * @return CData OrtRunOptions pointer
     *
     * @throws Exception on error
     */
    public function createRunOptions(): CData
    {
        $runOptions = $this->ffi->new('OrtRunOptions*');
        $status = ($this->api->CreateRunOptions)(\FFI::addr($runOptions));
        Lib::checkStatus($status);

        return $runOptions;
    }

    /**
     * Release run options.
     *
     * @param CData $runOptions OrtRunOptions pointer
     */
    public function releaseRunOptions(CData $runOptions): void
    {
        ($this->api->ReleaseRunOptions)($runOptions);
    }

    /**
     * Set run options log verbosity level.
     *
     * @param CData        $runOptions OrtRunOptions pointer
     * @param LoggingLevel $level      Log verbosity level
     *
     * @throws Exception on error
     */
    public function setRunOptionsLogVerbosityLevel(CData $runOptions, LoggingLevel $level): void
    {
        $status = ($this->api->RunOptionsSetRunLogVerbosityLevel)($runOptions, $level->value);
        Lib::checkStatus($status);
    }

    /**
     * Set run options log severity level.
     *
     * @param CData        $runOptions OrtRunOptions pointer
     * @param LoggingLevel $level      Log severity level
     *
     * @throws Exception on error
     */
    public function setRunOptionsLogSeverityLevel(CData $runOptions, LoggingLevel $level): void
    {
        $status = ($this->api->RunOptionsSetRunLogSeverityLevel)($runOptions, $level->value);
        Lib::checkStatus($status);
    }

    /**
     * Set run options tag.
     *
     * @param CData  $runOptions OrtRunOptions pointer
     * @param string $tag        Run tag for profiling/identification
     *
     * @throws Exception on error
     */
    public function setRunOptionsRunTag(CData $runOptions, string $tag): void
    {
        $status = ($this->api->RunOptionsSetRunTag)($runOptions, $tag);
        Lib::checkStatus($status);
    }

    /**
     * Set run options terminate flag.
     *
     * @param CData $runOptions OrtRunOptions pointer
     * @param bool  $terminate  Whether to request termination
     *
     * @throws Exception on error
     */
    public function setRunOptionsTerminate(CData $runOptions, bool $terminate): void
    {
        if ($terminate) {
            $status = ($this->api->RunOptionsSetTerminate)($runOptions);
        } else {
            $status = ($this->api->RunOptionsUnsetTerminate)($runOptions);
        }
        Lib::checkStatus($status);
    }

    /**
     * Run inference.
     *
     * @param CData           $session     OrtSession pointer
     * @param RunOptions      $runOptions  Run options
     * @param array           $inputNames  Array of input name strings
     * @param array<OrtValue> $inputs      Array of OrtValues
     * @param array           $outputNames Array of output name strings
     *
     * @return array<OrtValue> Array of OrtValues
     *
     * @throws Exception on error
     */
    public function run(
        CData $session,
        RunOptions $runOptions,
        array $inputNames,
        array $inputs,
        array $outputNames
    ): array {
        $inputCount = \count($inputs);
        $outputCount = \count($outputNames);

        $inputNamesC = $this->ffi->new("const char*[{$inputCount}]");
        foreach ($inputNames as $i => $name) {
            $inputNamesC[$i] = $this->ffi->new('char['.(\strlen($name) + 1).']', false);
            \FFI::memcpy($inputNamesC[$i], $name, \strlen($name));
        }

        $outputNamesC = $this->ffi->new("const char*[{$outputCount}]");
        foreach ($outputNames as $i => $name) {
            $outputNamesC[$i] = $this->ffi->new('char['.(\strlen($name) + 1).']', false);
            \FFI::memcpy($outputNamesC[$i], $name, \strlen($name));
        }

        $inputsC = $this->ffi->new("OrtValue*[{$inputCount}]");
        foreach ($inputs as $i => $input) {
            $inputsC[$i] = $input->handle;
        }

        $outputs = $this->ffi->new("OrtValue*[{$outputCount}]");

        $status = ($this->api->Run)(
            $session,
            $runOptions->getHandle(),
            $inputNamesC,
            $inputsC,
            $inputCount,
            $outputNamesC,
            $outputCount,
            $outputs
        );

        for ($i = 0; $i < $inputCount; ++$i) {
            \FFI::free($inputNamesC[$i]);
        }
        for ($i = 0; $i < $outputCount; ++$i) {
            \FFI::free($outputNamesC[$i]);
        }

        Lib::checkStatus($status);

        $result = [];
        for ($i = 0; $i < $outputCount; ++$i) {
            $result[] = OrtValue::fromHandle($outputs[$i]);
        }

        return $result;
    }

    /**
     * Get default allocator.
     *
     * @return CData OrtAllocator pointer
     *
     * @throws Exception on error
     */
    public function getAllocatorWithDefaultOptions(): CData
    {
        $allocator = $this->ffi->new('OrtAllocator*');
        $status = ($this->api->GetAllocatorWithDefaultOptions)(\FFI::addr($allocator));
        Lib::checkStatus($status);

        return $allocator;
    }

    /**
     * Append CUDA execution provider.
     *
     * @param CData $options  OrtSessionOptions pointer
     * @param int   $deviceId CUDA device ID
     *
     * @throws Exception on error
     */
    public function sessionOptionsAppendExecutionProviderCUDA(CData $options, int $deviceId): void
    {
        $status = ($this->api->SessionOptionsAppendExecutionProvider_CUDA)($options, $deviceId);
        Lib::checkStatus($status);
    }

    /**
     * Append CoreML execution provider.
     *
     * @param CData $options OrtSessionOptions pointer
     * @param int   $flags   CoreML flags
     *
     * @throws Exception on error
     */
    public function sessionOptionsAppendExecutionProviderCoreML(CData $options, int $flags): void
    {
        $status = $this->ffi->OrtSessionOptionsAppendExecutionProvider_CoreML($options, $flags);
        Lib::checkStatus($status);
    }

    /**
     * Append DirectML execution provider.
     *
     * @param CData $options  OrtSessionOptions pointer
     * @param int   $deviceId DirectML device ID
     *
     * @throws Exception on error
     */
    public function sessionOptionsAppendExecutionProviderDirectML(CData $options, int $deviceId): void
    {
        $status = ($this->api->SessionOptionsAppendExecutionProvider_DirectML)($options, $deviceId);
        Lib::checkStatus($status);
    }

    /**
     * Append DirectML execution provider using DML2 API.
     *
     * @param CData $options               OrtSessionOptions pointer
     * @param int   $performancePreference Performance preference (0=Default, 1=HighPerformance, 2=MinimumPower)
     * @param int   $deviceFilter          Device filter bitmask (1=GPU)
     *
     * @throws Exception on error
     */
    public function sessionOptionsAppendExecutionProviderDML2(CData $options, int $performancePreference, int $deviceFilter): void
    {
        // Get the DML API
        $dmlApi = $this->getDmlApi();

        // Create device options struct
        $deviceOpts = $this->ffi->new('OrtDmlDeviceOptions');
        $deviceOpts->Preference = $performancePreference;
        $deviceOpts->Filter = $deviceFilter;

        // Call DML2 function
        $status = ($dmlApi->SessionOptionsAppendExecutionProvider_DML2)($options, \FFI::addr($deviceOpts));
        Lib::checkStatus($status);
    }

    /**
     * Get the DirectML API struct.
     *
     * @return CData OrtDmlApi pointer
     *
     * @throws Exception if DML API is not available
     */
    private function getDmlApi(): CData
    {
        static $dmlApi = null;

        if (null === $dmlApi) {
            $dmlApiPtr = $this->ffi->new('OrtDmlApi*');

            // GetExecutionProviderApi should be in the main OrtApi struct
            if (!isset($this->api->GetExecutionProviderApi)) {
                throw new \RuntimeException('GetExecutionProviderApi not available in this ONNX Runtime version');
            }

            $status = ($this->api->GetExecutionProviderApi)('DML', \FFI::addr($dmlApiPtr));
            Lib::checkStatus($status);

            $dmlApi = $dmlApiPtr;
        }

        return $dmlApi;
    }

    /**
     * Append generic execution provider.
     *
     * @param CData                $options   OrtSessionOptions pointer
     * @param string               $name      Provider name
     * @param array<string,string> $keyValues Provider options as key-value pairs
     *
     * @throws Exception on error
     */
    public function sessionOptionsAppendExecutionProvider(CData $options, string $name, array $keyValues = []): void
    {
        $numKeys = \count($keyValues);
        
        // Always create arrays even if empty - passing null can cause segfaults
        $keysPtr = $numKeys > 0 ? $this->ffi->new('char*[' . $numKeys . ']') : $this->ffi->new('char*[1]');
        $valuesPtr = $numKeys > 0 ? $this->ffi->new('char*[' . $numKeys . ']') : $this->ffi->new('char*[1]');

        if ($numKeys > 0) {
            $keys = [];
            $values = [];

            foreach ($keyValues as $key => $value) {
                $keys[] = $key;
                $values[] = $value;
            }

            // Keep references to prevent GC
            $keyArrays = [];
            $valueArrays = [];

            for ($i = 0; $i < $numKeys; ++$i) {
                $keyArrays[$i] = $this->ffi->new('char[' . (\strlen($keys[$i]) + 1) . ']');
                \FFI::memcpy($keyArrays[$i], $keys[$i], \strlen($keys[$i]));
                $keyArrays[$i][\strlen($keys[$i])] = "\0";
                $keysPtr[$i] = $this->ffi->cast('char*', $keyArrays[$i]);

                $valueArrays[$i] = $this->ffi->new('char[' . (\strlen($values[$i]) + 1) . ']');
                \FFI::memcpy($valueArrays[$i], $values[$i], \strlen($values[$i]));
                $valueArrays[$i][\strlen($values[$i])] = "\0";
                $valuesPtr[$i] = $this->ffi->cast('char*', $valueArrays[$i]);
            }
        }

        $status = ($this->api->SessionOptionsAppendExecutionProvider)(
            $options,
            $name,
            $keysPtr,
            $valuesPtr,
            $numKeys
        );

        Lib::checkStatus($status);
    }

    // ==================== CUDA V2 API ====================

    /**
     * Create CUDA provider options.
     *
     * @return CData OrtCUDAProviderOptionsV2 pointer
     *
     * @throws Exception on error
     */
    public function createCUDAProviderOptions(): CData
    {
        $options = $this->ffi->new('OrtCUDAProviderOptionsV2*');
        $status = ($this->api->CreateCUDAProviderOptions)(\FFI::addr($options));
        Lib::checkStatus($status);

        return $options;
    }

    /**
     * Update CUDA provider options.
     *
     * @param CData                $options   OrtCUDAProviderOptionsV2 pointer
     * @param array<string,string> $keyValues Key-value pairs of options
     *
     * @throws Exception on error
     */
    public function updateCUDAProviderOptions(CData $options, array $keyValues): void
    {
        $numKeys = \count($keyValues);

        if (0 === $numKeys) {
            return;
        }

        $keys = [];
        $values = [];

        foreach ($keyValues as $key => $value) {
            $keys[] = $key;
            $values[] = $value;
        }

        $keysPtr = $this->ffi->new('char*[' . $numKeys . ']');
        $valuesPtr = $this->ffi->new('char*[' . $numKeys . ']');

        for ($i = 0; $i < $numKeys; ++$i) {
            $keysPtr[$i] = $this->ffi->new('char[' . (\strlen($keys[$i]) + 1) . ']');
            \FFI::memcpy($keysPtr[$i], $keys[$i], \strlen($keys[$i]));

            $valuesPtr[$i] = $this->ffi->new('char[' . (\strlen($values[$i]) + 1) . ']');
            \FFI::memcpy($valuesPtr[$i], $values[$i], \strlen($values[$i]));
        }

        $status = ($this->api->UpdateCUDAProviderOptions)(
            $options,
            $keysPtr,
            $valuesPtr,
            $numKeys
        );

        Lib::checkStatus($status);
    }

    /**
     * Release CUDA provider options.
     *
     * @param CData $options OrtCUDAProviderOptionsV2 pointer
     */
    public function releaseCUDAProviderOptions(CData $options): void
    {
        ($this->api->ReleaseCUDAProviderOptions)($options);
    }

    /**
     * Append CUDA execution provider using V2 API.
     *
     * @param CData $sessionOptions OrtSessionOptions pointer
     * @param CData $cudaOptions    OrtCUDAProviderOptionsV2 pointer
     *
     * @throws Exception on error
     */
    public function sessionOptionsAppendExecutionProviderCUDA_V2(CData $sessionOptions, CData $cudaOptions): void
    {
        $status = ($this->api->SessionOptionsAppendExecutionProvider_CUDA_V2)($sessionOptions, $cudaOptions);
        Lib::checkStatus($status);
    }

    // ==================== TensorRT V2 API ====================

    /**
     * Create TensorRT provider options.
     *
     * @return CData OrtTensorRTProviderOptionsV2 pointer
     *
     * @throws Exception on error
     */
    public function createTensorRTProviderOptions(): CData
    {
        $options = $this->ffi->new('OrtTensorRTProviderOptionsV2*');
        $status = ($this->api->CreateTensorRTProviderOptions)(\FFI::addr($options));
        Lib::checkStatus($status);

        return $options;
    }

    /**
     * Update TensorRT provider options.
     *  
     * @param CData                $options   OrtTensorRTProviderOptionsV2 pointer
     * @param array<string,string> $keyValues Key-value pairs of options
     *
     * @throws Exception on error
     */
    public function updateTensorRTProviderOptions(CData $options, array $keyValues): void
    {
        $numKeys = \count($keyValues);

        if (0 === $numKeys) {
            return;
        }

        $keys = [];
        $values = [];

        foreach ($keyValues as $key => $value) {
            $keys[] = $key;
            $values[] = $value;
        }

        $keysPtr = $this->ffi->new('char*[' . $numKeys . ']');
        $valuesPtr = $this->ffi->new('char*[' . $numKeys . ']');

        for ($i = 0; $i < $numKeys; ++$i) {
            $keysPtr[$i] = $this->ffi->new('char[' . (\strlen($keys[$i]) + 1) . ']');
            \FFI::memcpy($keysPtr[$i], $keys[$i], \strlen($keys[$i]));

            $valuesPtr[$i] = $this->ffi->new('char[' . (\strlen($values[$i]) + 1) . ']');
            \FFI::memcpy($valuesPtr[$i], $values[$i], \strlen($values[$i]));
        }

        $status = ($this->api->UpdateTensorRTProviderOptions)(
            $options,
            $keysPtr,
            $valuesPtr,
            $numKeys
        );

        Lib::checkStatus($status);
    }

    /**
     * Release TensorRT provider options.
     *
     * @param CData $options OrtTensorRTProviderOptionsV2 pointer
     */
    public function releaseTensorRTProviderOptions(CData $options): void
    {
        ($this->api->ReleaseTensorRTProviderOptions)($options);
    }

    /**
     * Append TensorRT execution provider using V2 API.
     *
     * @param CData $sessionOptions OrtSessionOptions pointer
     * @param CData $tensorrtOptions OrtTensorRTProviderOptionsV2 pointer
     *
     * @throws Exception on error
     */
    public function sessionOptionsAppendExecutionProviderTensorRT_V2(CData $sessionOptions, CData $tensorrtOptions): void
    {
        $status = ($this->api->SessionOptionsAppendExecutionProvider_TensorRT_V2)($sessionOptions, $tensorrtOptions);
        Lib::checkStatus($status);
    }

    /**
     * Disable telemetry events.
     *
     * Privacy info: https://github.com/microsoft/onnxruntime/blob/main/docs/Privacy.md
     *
     * @param CData $env OrtEnv pointer
     *
     * @throws Exception on error
     */
    public function disableTelemetryEvents(CData $env): void
    {
        $status = ($this->api->DisableTelemetryEvents)($env);
        Lib::checkStatus($status);
    }

    /**
     * Get error message from status.
     *
     * @param CData $status OrtStatus pointer
     *
     * @return string Error message
     */
    public function getErrorMessage(CData $status): string
    {
        $message = ($this->api->GetErrorMessage)($status);

        return \is_string($message) ? $message : \FFI::string($message);
    }

    /**
     * Get error code from status.
     *
     * @param CData $status OrtStatus pointer
     *
     * @return int Error code (OrtErrorCode enum value)
     */
    public function getErrorCode(CData $status): int
    {
        return ($this->api->GetErrorCode)($status);
    }

    /**
     * Release status.
     *
     * @param CData $status OrtStatus pointer
     */
    public function releaseStatus(CData $status): void
    {
        ($this->api->ReleaseStatus)($status);
    }

    /**
     * Get the version string of the ONNX Runtime library.
     *
     * @return string Version string
     */
    public function getVersionString(): string
    {
        return ($this->api->GetVersionString)();
    }

    /**
     * Get the ONNX type of an OrtValue.
     *
     * @param CData $value OrtValue pointer
     *
     * @return int ONNXType enum value
     *
     * @throws Exception on error
     */
    public function getValueType(CData $value): int
    {
        $type = $this->ffi->new('ONNXType');
        $status = ($this->api->GetValueType)($value, \FFI::addr($type));
        Lib::checkStatus($status);

        return $type->cdata;
    }

    /**
     * Create a composite value (sequence, map, or optional).
     *
     * @param array    $values Array of OrtValue pointers
     * @param OnnxType $type   Value type
     *
     * @return CData OrtValue pointer
     *
     * @throws Exception on error
     */
    public function createValue(array $values, OnnxType $type): CData
    {
        $count = \count($values);
        $valuesArray = $this->ffi->new("OrtValue*[{$count}]");
        foreach ($values as $i => $value) {
            $valuesArray[$i] = $value;
        }

        $value = $this->ffi->new('OrtValue*');
        $status = ($this->api->CreateValue)(
            $valuesArray,
            $count,
            $type->value,
            \FFI::addr($value)
        );
        Lib::checkStatus($status);

        return $value;
    }

    /**
     * Get count of values in sequence or map.
     *
     * @param CData $value OrtValue pointer (sequence or map)
     *
     * @return int Number of values
     *
     * @throws Exception on error
     */
    public function getValueCount(CData $value): int
    {
        $count = $this->ffi->new('size_t');
        $status = ($this->api->GetValueCount)($value, \FFI::addr($count));
        Lib::checkStatus($status);

        return $count->cdata;
    }

    /**
     * Get value from sequence or map by index.
     *
     * @param CData $value     OrtValue pointer (sequence or map)
     * @param int   $index     Element index
     * @param CData $allocator OrtAllocator pointer
     *
     * @return CData OrtValue pointer
     *
     * @throws Exception on error
     */
    public function getValue(CData $value, int $index, CData $allocator): CData
    {
        $element = $this->ffi->new('OrtValue*');
        $status = ($this->api->GetValue)(
            $value,
            $index,
            $allocator,
            \FFI::addr($element)
        );
        Lib::checkStatus($status);

        return $element;
    }

    /**
     * Get tensor size in bytes.
     *
     * @param CData $value OrtValue pointer (tensor)
     *
     * @return int Size in bytes
     *
     * @throws Exception on error
     */
    public function getTensorSizeInBytes(CData $value): int
    {
        $size = $this->ffi->new('size_t');
        $status = ($this->api->GetTensorSizeInBytes)($value, \FFI::addr($size));
        Lib::checkStatus($status);

        return $size->cdata;
    }

    /**
     * Create tensor without data (for strings or allocated tensors).
     *
     * @param CData $allocator OrtAllocator pointer
     * @param array $shape     Tensor shape
     * @param int   $type      Element type enum value
     *
     * @return CData OrtValue pointer
     *
     * @throws Exception on error
     */
    public function createTensorAsOrtValue(CData $allocator, array $shape, int $type): CData
    {
        $shapeSize = \count($shape);
        $shapeArray = $this->ffi->new("int64_t[{$shapeSize}]");
        foreach ($shape as $i => $dim) {
            $shapeArray[$i] = $dim;
        }

        $value = $this->ffi->new('OrtValue*');
        $status = ($this->api->CreateTensorAsOrtValue)(
            $allocator,
            $shapeArray,
            $shapeSize,
            $type,
            \FFI::addr($value)
        );
        Lib::checkStatus($status);

        return $value;
    }

    /**
     * Fill string tensor elements.
     *
     * @param CData $value   OrtValue pointer (string tensor)
     * @param array $strings Array of strings to fill
     *
     * @throws Exception on error
     */
    public function fillStringTensor(CData $value, array $strings): void
    {
        $count = \count($strings);
        if (0 === $count) {
            return;
        }

        $strArray = $this->ffi->new("char*[{$count}]");

        foreach ($strings as $i => $str) {
            $len = \strlen($str);
            $cStr = $this->ffi->new('char['.($len + 1).']', false);
            \FFI::memcpy($cStr, $str, $len);
            $cStr[$len] = "\0";
            $strArray[$i] = $this->ffi->cast('char*', $cStr);
        }

        $status = ($this->api->FillStringTensor)($value, $strArray, $count);
        Lib::checkStatus($status);
    }

    /**
     * Get total data length for string tensor.
     *
     * @param CData $value OrtValue pointer (string tensor)
     *
     * @return int Total length in bytes
     *
     * @throws Exception on error
     */
    public function getStringTensorDataLength(CData $value): int
    {
        $length = $this->ffi->new('size_t');
        $status = ($this->api->GetStringTensorDataLength)(
            $value,
            \FFI::addr($length)
        );
        Lib::checkStatus($status);

        return $length->cdata;
    }

    /**
     * Get string tensor content with offsets.
     *
     * @param CData $value OrtValue pointer (string tensor)
     * @param int   $count Number of elements
     *
     * @return array Array of strings
     *
     * @throws Exception on error
     */
    public function getStringTensorContent(CData $value, int $count): array
    {
        $dataLength = $this->getStringTensorDataLength($value);

        $data = $this->ffi->new("char[{$dataLength}]");
        $offsets = $this->ffi->new("size_t[{$count}]");

        $status = ($this->api->GetStringTensorContent)(
            $value,
            $data,
            $dataLength,
            $offsets,
            $count
        );
        Lib::checkStatus($status);

        $strings = [];
        for ($i = 0; $i < $count; ++$i) {
            $start = $offsets[$i];
            $end = ($i < $count - 1) ? $offsets[$i + 1] : $dataLength;
            $len = $end - $start;
            $stringPtr = $data + $start;

            $strings[] = \FFI::string($stringPtr, $len);
        }

        return $strings;
    }

    /**
     * Get available execution providers.
     *
     * Returns a list of execution provider names that are available in the current
     * ONNX Runtime installation. This can be used to check if GPU support (CUDA, DirectML)
     * is available before attempting to use it.
     *
     * @return string[] Array of available provider names (e.g., ['CPUExecutionProvider', 'CUDAExecutionProvider'])
     *
     * @throws Exception on error
     */
    public function getAvailableProviders(): array
    {
        $providersPtr = $this->ffi->new('char**');
        $length = $this->ffi->new('int');

        $status = ($this->api->GetAvailableProviders)(\FFI::addr($providersPtr), \FFI::addr($length));
        Lib::checkStatus($status);

        $providers = [];
        $count = $length->cdata;

        for ($i = 0; $i < $count; ++$i) {
            $providers[] = \FFI::string($providersPtr[$i]);
        }

        $status = ($this->api->ReleaseAvailableProviders)($providersPtr, $count);
        Lib::checkStatus($status);

        return $providers;
    }
}

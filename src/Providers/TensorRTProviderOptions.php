<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Providers;

use FFI\CData;
use PhpMlKit\ONNXRuntime\Enums\ExecutionProvider;
use PhpMlKit\ONNXRuntime\FFI\Lib;

/**
 * TensorRT execution provider options for NVIDIA GPUs.
 */
final class TensorRTProviderOptions extends V2ProviderOptions
{
    /**
     * @param int $deviceId CUDA device ID (0 = default)
     * @param int $trtMaxPartitionIterations Maximum iterations for TensorRT parser
     * @param int $trtMinSubgraphSize Minimum subgraph size for TensorRT
     * @param int $trtMaxWorkspaceSize Maximum workspace size in bytes
     * @param bool $trtFp16Enable Enable FP16 precision
     * @param bool $trtInt8Enable Enable INT8 precision
     * @param null|string $trtInt8CalibrationTableName INT8 calibration table path
     * @param bool $trtInt8UseNativeCalibrationTable Use native TensorRT calibration
     * @param bool $trtDlaEnable Enable DLA (Deep Learning Accelerator)
     * @param int $trtDlaCore DLA core to use (0 = default)
     * @param bool $trtDumpSubgraphs Dump TensorRT subgraphs
     * @param bool $trtEngineCacheEnable Enable engine caching
     * @param null|string $trtEngineCachePath Path for cached engines
     * @param bool $trtEngineDecryptionEnable Enable engine decryption
     * @param null|string $trtEngineDecryptionLibPath Path to decryption library
     * @param bool $trtForceSequentialEngineBuild Force sequential engine building
     */
    public function __construct(
        public readonly int $deviceId = 0,
        public readonly int $trtMaxPartitionIterations = 1000,
        public readonly int $trtMinSubgraphSize = 1,
        public readonly int $trtMaxWorkspaceSize = 1073741824, // 1GB default
        public readonly bool $trtFp16Enable = false,
        public readonly bool $trtInt8Enable = false,
        public readonly ?string $trtInt8CalibrationTableName = null,
        public readonly bool $trtInt8UseNativeCalibrationTable = false,
        public readonly bool $trtDlaEnable = false,
        public readonly int $trtDlaCore = 0,
        public readonly bool $trtDumpSubgraphs = false,
        public readonly bool $trtEngineCacheEnable = false,
        public readonly ?string $trtEngineCachePath = null,
        public readonly bool $trtEngineDecryptionEnable = false,
        public readonly ?string $trtEngineDecryptionLibPath = null,
        public readonly bool $trtForceSequentialEngineBuild = false,
    ) {}

    public function getProvider(): ExecutionProvider
    {
        return ExecutionProvider::TENSORRT;
    }

    public function toArray(): array
    {
        return [
            'device_id' => (string) $this->deviceId,
            'trt_max_partition_iterations' => (string) $this->trtMaxPartitionIterations,
            'trt_min_subgraph_size' => (string) $this->trtMinSubgraphSize,
            'trt_max_workspace_size' => (string) $this->trtMaxWorkspaceSize,
            'trt_fp16_enable' => $this->trtFp16Enable ? '1' : '0',
            'trt_int8_enable' => $this->trtInt8Enable ? '1' : '0',
            'trt_int8_calibration_table_name' => $this->trtInt8CalibrationTableName,
            'trt_int8_use_native_calibration_table' => $this->trtInt8UseNativeCalibrationTable ? '1' : '0',
            'trt_dla_enable' => $this->trtDlaEnable ? '1' : '0',
            'trt_dla_core' => (string) $this->trtDlaCore,
            'trt_dump_subgraphs' => $this->trtDumpSubgraphs ? '1' : '0',
            'trt_engine_cache_enable' => $this->trtEngineCacheEnable ? '1' : '0',
            'trt_engine_cache_path' => $this->trtEngineCachePath,
            'trt_engine_decryption_enable' => $this->trtEngineDecryptionEnable ? '1' : '0',
            'trt_engine_decryption_lib_path' => $this->trtEngineDecryptionLibPath,
            'trt_force_sequential_engine_build' => $this->trtForceSequentialEngineBuild ? '1' : '0',
        ];
    }

    public function clone(): static
    {
        return new self(
            deviceId: $this->deviceId,
            trtMaxPartitionIterations: $this->trtMaxPartitionIterations,
            trtMinSubgraphSize: $this->trtMinSubgraphSize,
            trtMaxWorkspaceSize: $this->trtMaxWorkspaceSize,
            trtFp16Enable: $this->trtFp16Enable,
            trtInt8Enable: $this->trtInt8Enable,
            trtInt8CalibrationTableName: $this->trtInt8CalibrationTableName,
            trtInt8UseNativeCalibrationTable: $this->trtInt8UseNativeCalibrationTable,
            trtDlaEnable: $this->trtDlaEnable,
            trtDlaCore: $this->trtDlaCore,
            trtDumpSubgraphs: $this->trtDumpSubgraphs,
            trtEngineCacheEnable: $this->trtEngineCacheEnable,
            trtEngineCachePath: $this->trtEngineCachePath,
            trtEngineDecryptionEnable: $this->trtEngineDecryptionEnable,
            trtEngineDecryptionLibPath: $this->trtEngineDecryptionLibPath,
            trtForceSequentialEngineBuild: $this->trtForceSequentialEngineBuild,
        );
    }

    /**
     * Create with default settings.
     */
    public static function default(): self
    {
        return new self();
    }

    /**
     * Create with engine caching enabled.
     */
    public static function withCache(string $cachePath): self
    {
        return new self(
            trtEngineCacheEnable: true,
            trtEngineCachePath: $cachePath,
        );
    }

    /**
     * Create with maximum performance settings.
     */
    public static function maximumPerformance(): self
    {
        return new self(
            trtFp16Enable: true,
            trtMaxWorkspaceSize: 4 * 1073741824, // 4GB
        );
    }

    public function withDeviceId(int $deviceId): self
    {
        return new self(
            deviceId: $deviceId,
            trtMaxPartitionIterations: $this->trtMaxPartitionIterations,
            trtMinSubgraphSize: $this->trtMinSubgraphSize,
            trtMaxWorkspaceSize: $this->trtMaxWorkspaceSize,
            trtFp16Enable: $this->trtFp16Enable,
            trtInt8Enable: $this->trtInt8Enable,
            trtInt8CalibrationTableName: $this->trtInt8CalibrationTableName,
            trtInt8UseNativeCalibrationTable: $this->trtInt8UseNativeCalibrationTable,
            trtDlaEnable: $this->trtDlaEnable,
            trtDlaCore: $this->trtDlaCore,
            trtDumpSubgraphs: $this->trtDumpSubgraphs,
            trtEngineCacheEnable: $this->trtEngineCacheEnable,
            trtEngineCachePath: $this->trtEngineCachePath,
            trtEngineDecryptionEnable: $this->trtEngineDecryptionEnable,
            trtEngineDecryptionLibPath: $this->trtEngineDecryptionLibPath,
            trtForceSequentialEngineBuild: $this->trtForceSequentialEngineBuild,
        );
    }

    public function withMaxPartitionIterations(int $trtMaxPartitionIterations): self
    {
        return new self(
            deviceId: $this->deviceId,
            trtMaxPartitionIterations: $trtMaxPartitionIterations,
            trtMinSubgraphSize: $this->trtMinSubgraphSize,
            trtMaxWorkspaceSize: $this->trtMaxWorkspaceSize,
            trtFp16Enable: $this->trtFp16Enable,
            trtInt8Enable: $this->trtInt8Enable,
            trtInt8CalibrationTableName: $this->trtInt8CalibrationTableName,
            trtInt8UseNativeCalibrationTable: $this->trtInt8UseNativeCalibrationTable,
            trtDlaEnable: $this->trtDlaEnable,
            trtDlaCore: $this->trtDlaCore,
            trtDumpSubgraphs: $this->trtDumpSubgraphs,
            trtEngineCacheEnable: $this->trtEngineCacheEnable,
            trtEngineCachePath: $this->trtEngineCachePath,
            trtEngineDecryptionEnable: $this->trtEngineDecryptionEnable,
            trtEngineDecryptionLibPath: $this->trtEngineDecryptionLibPath,
            trtForceSequentialEngineBuild: $this->trtForceSequentialEngineBuild,
        );
    }

    public function withMinSubgraphSize(int $trtMinSubgraphSize): self
    {
        return new self(
            deviceId: $this->deviceId,
            trtMaxPartitionIterations: $this->trtMaxPartitionIterations,
            trtMinSubgraphSize: $trtMinSubgraphSize,
            trtMaxWorkspaceSize: $this->trtMaxWorkspaceSize,
            trtFp16Enable: $this->trtFp16Enable,
            trtInt8Enable: $this->trtInt8Enable,
            trtInt8CalibrationTableName: $this->trtInt8CalibrationTableName,
            trtInt8UseNativeCalibrationTable: $this->trtInt8UseNativeCalibrationTable,
            trtDlaEnable: $this->trtDlaEnable,
            trtDlaCore: $this->trtDlaCore,
            trtDumpSubgraphs: $this->trtDumpSubgraphs,
            trtEngineCacheEnable: $this->trtEngineCacheEnable,
            trtEngineCachePath: $this->trtEngineCachePath,
            trtEngineDecryptionEnable: $this->trtEngineDecryptionEnable,
            trtEngineDecryptionLibPath: $this->trtEngineDecryptionLibPath,
            trtForceSequentialEngineBuild: $this->trtForceSequentialEngineBuild,
        );
    }

    public function withMaxWorkspaceSize(int $trtMaxWorkspaceSize): self
    {
        return new self(
            deviceId: $this->deviceId,
            trtMaxPartitionIterations: $this->trtMaxPartitionIterations,
            trtMinSubgraphSize: $this->trtMinSubgraphSize,
            trtMaxWorkspaceSize: $trtMaxWorkspaceSize,
            trtFp16Enable: $this->trtFp16Enable,
            trtInt8Enable: $this->trtInt8Enable,
            trtInt8CalibrationTableName: $this->trtInt8CalibrationTableName,
            trtInt8UseNativeCalibrationTable: $this->trtInt8UseNativeCalibrationTable,
            trtDlaEnable: $this->trtDlaEnable,
            trtDlaCore: $this->trtDlaCore,
            trtDumpSubgraphs: $this->trtDumpSubgraphs,
            trtEngineCacheEnable: $this->trtEngineCacheEnable,
            trtEngineCachePath: $this->trtEngineCachePath,
            trtEngineDecryptionEnable: $this->trtEngineDecryptionEnable,
            trtEngineDecryptionLibPath: $this->trtEngineDecryptionLibPath,
            trtForceSequentialEngineBuild: $this->trtForceSequentialEngineBuild,
        );
    }

    public function withFp16(bool $trtFp16Enable = true): self
    {
        return new self(
            deviceId: $this->deviceId,
            trtMaxPartitionIterations: $this->trtMaxPartitionIterations,
            trtMinSubgraphSize: $this->trtMinSubgraphSize,
            trtMaxWorkspaceSize: $this->trtMaxWorkspaceSize,
            trtFp16Enable: $trtFp16Enable,
            trtInt8Enable: $this->trtInt8Enable,
            trtInt8CalibrationTableName: $this->trtInt8CalibrationTableName,
            trtInt8UseNativeCalibrationTable: $this->trtInt8UseNativeCalibrationTable,
            trtDlaEnable: $this->trtDlaEnable,
            trtDlaCore: $this->trtDlaCore,
            trtDumpSubgraphs: $this->trtDumpSubgraphs,
            trtEngineCacheEnable: $this->trtEngineCacheEnable,
            trtEngineCachePath: $this->trtEngineCachePath,
            trtEngineDecryptionEnable: $this->trtEngineDecryptionEnable,
            trtEngineDecryptionLibPath: $this->trtEngineDecryptionLibPath,
            trtForceSequentialEngineBuild: $this->trtForceSequentialEngineBuild,
        );
    }

    public function withInt8(bool $trtInt8Enable = true, ?string $calibrationTablePath = null): self
    {
        return new self(
            deviceId: $this->deviceId,
            trtMaxPartitionIterations: $this->trtMaxPartitionIterations,
            trtMinSubgraphSize: $this->trtMinSubgraphSize,
            trtMaxWorkspaceSize: $this->trtMaxWorkspaceSize,
            trtFp16Enable: $this->trtFp16Enable,
            trtInt8Enable: $trtInt8Enable,
            trtInt8CalibrationTableName: $calibrationTablePath ?? $this->trtInt8CalibrationTableName,
            trtInt8UseNativeCalibrationTable: $this->trtInt8UseNativeCalibrationTable,
            trtDlaEnable: $this->trtDlaEnable,
            trtDlaCore: $this->trtDlaCore,
            trtDumpSubgraphs: $this->trtDumpSubgraphs,
            trtEngineCacheEnable: $this->trtEngineCacheEnable,
            trtEngineCachePath: $this->trtEngineCachePath,
            trtEngineDecryptionEnable: $this->trtEngineDecryptionEnable,
            trtEngineDecryptionLibPath: $this->trtEngineDecryptionLibPath,
            trtForceSequentialEngineBuild: $this->trtForceSequentialEngineBuild,
        );
    }

    public function withDla(bool $trtDlaEnable = true, int $trtDlaCore = 0): self
    {
        return new self(
            deviceId: $this->deviceId,
            trtMaxPartitionIterations: $this->trtMaxPartitionIterations,
            trtMinSubgraphSize: $this->trtMinSubgraphSize,
            trtMaxWorkspaceSize: $this->trtMaxWorkspaceSize,
            trtFp16Enable: $this->trtFp16Enable,
            trtInt8Enable: $this->trtInt8Enable,
            trtInt8CalibrationTableName: $this->trtInt8CalibrationTableName,
            trtInt8UseNativeCalibrationTable: $this->trtInt8UseNativeCalibrationTable,
            trtDlaEnable: $trtDlaEnable,
            trtDlaCore: $trtDlaCore,
            trtDumpSubgraphs: $this->trtDumpSubgraphs,
            trtEngineCacheEnable: $this->trtEngineCacheEnable,
            trtEngineCachePath: $this->trtEngineCachePath,
            trtEngineDecryptionEnable: $this->trtEngineDecryptionEnable,
            trtEngineDecryptionLibPath: $this->trtEngineDecryptionLibPath,
            trtForceSequentialEngineBuild: $this->trtForceSequentialEngineBuild,
        );
    }

    public function withDumpSubgraphs(bool $trtDumpSubgraphs = true): self
    {
        return new self(
            deviceId: $this->deviceId,
            trtMaxPartitionIterations: $this->trtMaxPartitionIterations,
            trtMinSubgraphSize: $this->trtMinSubgraphSize,
            trtMaxWorkspaceSize: $this->trtMaxWorkspaceSize,
            trtFp16Enable: $this->trtFp16Enable,
            trtInt8Enable: $this->trtInt8Enable,
            trtInt8CalibrationTableName: $this->trtInt8CalibrationTableName,
            trtInt8UseNativeCalibrationTable: $this->trtInt8UseNativeCalibrationTable,
            trtDlaEnable: $this->trtDlaEnable,
            trtDlaCore: $this->trtDlaCore,
            trtDumpSubgraphs: $trtDumpSubgraphs,
            trtEngineCacheEnable: $this->trtEngineCacheEnable,
            trtEngineCachePath: $this->trtEngineCachePath,
            trtEngineDecryptionEnable: $this->trtEngineDecryptionEnable,
            trtEngineDecryptionLibPath: $this->trtEngineDecryptionLibPath,
            trtForceSequentialEngineBuild: $this->trtForceSequentialEngineBuild,
        );
    }

    public function withEngineCache(bool $trtEngineCacheEnable = true, ?string $cachePath = null): self
    {
        return new self(
            deviceId: $this->deviceId,
            trtMaxPartitionIterations: $this->trtMaxPartitionIterations,
            trtMinSubgraphSize: $this->trtMinSubgraphSize,
            trtMaxWorkspaceSize: $this->trtMaxWorkspaceSize,
            trtFp16Enable: $this->trtFp16Enable,
            trtInt8Enable: $this->trtInt8Enable,
            trtInt8CalibrationTableName: $this->trtInt8CalibrationTableName,
            trtInt8UseNativeCalibrationTable: $this->trtInt8UseNativeCalibrationTable,
            trtDlaEnable: $this->trtDlaEnable,
            trtDlaCore: $this->trtDlaCore,
            trtDumpSubgraphs: $this->trtDumpSubgraphs,
            trtEngineCacheEnable: $trtEngineCacheEnable,
            trtEngineCachePath: $cachePath ?? $this->trtEngineCachePath,
            trtEngineDecryptionEnable: $this->trtEngineDecryptionEnable,
            trtEngineDecryptionLibPath: $this->trtEngineDecryptionLibPath,
            trtForceSequentialEngineBuild: $this->trtForceSequentialEngineBuild,
        );
    }

    public function withForceSequentialEngineBuild(bool $trtForceSequentialEngineBuild = true): self
    {
        return new self(
            deviceId: $this->deviceId,
            trtMaxPartitionIterations: $this->trtMaxPartitionIterations,
            trtMinSubgraphSize: $this->trtMinSubgraphSize,
            trtMaxWorkspaceSize: $this->trtMaxWorkspaceSize,
            trtFp16Enable: $this->trtFp16Enable,
            trtInt8Enable: $this->trtInt8Enable,
            trtInt8CalibrationTableName: $this->trtInt8CalibrationTableName,
            trtInt8UseNativeCalibrationTable: $this->trtInt8UseNativeCalibrationTable,
            trtDlaEnable: $this->trtDlaEnable,
            trtDlaCore: $this->trtDlaCore,
            trtDumpSubgraphs: $this->trtDumpSubgraphs,
            trtEngineCacheEnable: $this->trtEngineCacheEnable,
            trtEngineCachePath: $this->trtEngineCachePath,
            trtEngineDecryptionEnable: $this->trtEngineDecryptionEnable,
            trtEngineDecryptionLibPath: $this->trtEngineDecryptionLibPath,
            trtForceSequentialEngineBuild: $trtForceSequentialEngineBuild,
        );
    }

    protected function createOptions(): CData
    {
        return Lib::api()->createTensorRTProviderOptions();
    }

    protected function updateOptions(CData $options): void
    {
        $keyValues = [];

        // Only set non-default values to minimize overhead
        if (0 !== $this->deviceId) {
            $keyValues['device_id'] = (string) $this->deviceId;
        }

        if (1000 !== $this->trtMaxPartitionIterations) {
            $keyValues['trt_max_partition_iterations'] = (string) $this->trtMaxPartitionIterations;
        }

        if (1 !== $this->trtMinSubgraphSize) {
            $keyValues['trt_min_subgraph_size'] = (string) $this->trtMinSubgraphSize;
        }

        if (1073741824 !== $this->trtMaxWorkspaceSize) {
            $keyValues['trt_max_workspace_size'] = (string) $this->trtMaxWorkspaceSize;
        }

        if ($this->trtFp16Enable) {
            $keyValues['trt_fp16_enable'] = '1';
        }

        if ($this->trtInt8Enable) {
            $keyValues['trt_int8_enable'] = '1';
        }

        if (null !== $this->trtInt8CalibrationTableName) {
            $keyValues['trt_int8_calibration_table_name'] = $this->trtInt8CalibrationTableName;
        }

        if ($this->trtInt8UseNativeCalibrationTable) {
            $keyValues['trt_int8_use_native_calibration_table'] = '1';
        }

        if ($this->trtDlaEnable) {
            $keyValues['trt_dla_enable'] = '1';
        }

        if (0 !== $this->trtDlaCore) {
            $keyValues['trt_dla_core'] = (string) $this->trtDlaCore;
        }

        if ($this->trtDumpSubgraphs) {
            $keyValues['trt_dump_subgraphs'] = '1';
        }

        if ($this->trtEngineCacheEnable) {
            $keyValues['trt_engine_cache_enable'] = '1';
        }

        if (null !== $this->trtEngineCachePath) {
            $keyValues['trt_engine_cache_path'] = $this->trtEngineCachePath;
        }

        if ($this->trtEngineDecryptionEnable) {
            $keyValues['trt_engine_decryption_enable'] = '1';
        }

        if (null !== $this->trtEngineDecryptionLibPath) {
            $keyValues['trt_engine_decryption_lib_path'] = $this->trtEngineDecryptionLibPath;
        }

        if ($this->trtForceSequentialEngineBuild) {
            $keyValues['trt_force_sequential_engine_build'] = '1';
        }

        if (!empty($keyValues)) {
            Lib::api()->updateTensorRTProviderOptions($options, $keyValues);
        }
    }

    protected function appendProvider(CData $sessionOptions, CData $providerOptions): void
    {
        Lib::api()->sessionOptionsAppendExecutionProviderTensorRT_V2($sessionOptions, $providerOptions);
    }

    protected function releaseOptions(CData $options): void
    {
        Lib::api()->releaseTensorRTProviderOptions($options);
    }
}

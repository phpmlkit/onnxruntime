<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Providers;

use FFI\CData;
use PhpMlKit\ONNXRuntime\Enums\ArenaExtendStrategy;
use PhpMlKit\ONNXRuntime\Enums\CudnnConvAlgoSearch;
use PhpMlKit\ONNXRuntime\Enums\ExecutionProvider;
use PhpMlKit\ONNXRuntime\FFI\Lib;

/**
 * CUDA execution provider options.
 */
final class CudaProviderOptions extends V2ProviderOptions
{
    /**
     * @param int                      $deviceId                     CUDA device ID (0 = default)
     * @param null|int                 $gpuMemoryLimit               Maximum GPU memory in bytes (null = unlimited)
     * @param null|CudnnConvAlgoSearch $cudnnConvAlgoSearch          CUDNN convolution algorithm search
     * @param null|ArenaExtendStrategy $arenaExtendStrategy          Memory arena extension strategy
     * @param bool                     $doCopyInDefaultStream        Use same stream for copy and compute
     * @param bool                     $tunableOpEnable              Enable TunableOp
     * @param bool                     $tunableOpTuningEnable        Enable TunableOp tuning
     * @param int                      $tunableOpMaxTuningDurationMs Max tuning duration per op (0 = unlimited)
     */
    public function __construct(
        public readonly int $deviceId = 0,
        public readonly ?int $gpuMemoryLimit = null,
        public readonly ?CudnnConvAlgoSearch $cudnnConvAlgoSearch = null,
        public readonly ?ArenaExtendStrategy $arenaExtendStrategy = null,
        public readonly bool $doCopyInDefaultStream = true,
        public readonly bool $tunableOpEnable = false,
        public readonly bool $tunableOpTuningEnable = false,
        public readonly int $tunableOpMaxTuningDurationMs = 0,
    ) {}

    public function getProvider(): ExecutionProvider
    {
        return ExecutionProvider::CUDA;
    }

    public function toArray(): array
    {
        return [
            'device_id' => (string) $this->deviceId,
            'gpu_mem_limit' => null !== $this->gpuMemoryLimit ? (string) $this->gpuMemoryLimit : null,
            'cudnn_conv_algo_search' => $this->cudnnConvAlgoSearch?->value,
            'arena_extend_strategy' => $this->arenaExtendStrategy?->value,
            'do_copy_in_default_stream' => $this->doCopyInDefaultStream ? '1' : '0',
            'tunable_op_enable' => $this->tunableOpEnable ? '1' : '0',
            'tunable_op_tuning_enable' => $this->tunableOpTuningEnable ? '1' : '0',
            'tunable_op_max_tuning_duration_ms' => (string) $this->tunableOpMaxTuningDurationMs,
        ];
    }

    /**
     * Create with default settings.
     */
    public static function default(): self
    {
        return new self();
    }

    /**
     * Create with default high-performance settings.
     */
    public static function highPerformance(): self
    {
        return new self(
            cudnnConvAlgoSearch: CudnnConvAlgoSearch::HEURISTIC,
            arenaExtendStrategy: ArenaExtendStrategy::SAME_AS_REQUESTED,
        );
    }

    /**
     * Create with conservative memory usage.
     */
    public static function memoryConservative(int $memoryLimitBytes = 2 * 1024 * 1024 * 1024): self
    {
        return new self(
            gpuMemoryLimit: $memoryLimitBytes,
            arenaExtendStrategy: ArenaExtendStrategy::SAME_AS_REQUESTED,
        );
    }

    public function withDeviceId(int $deviceId): self
    {
        return new self(
            deviceId: $deviceId,
            gpuMemoryLimit: $this->gpuMemoryLimit,
            cudnnConvAlgoSearch: $this->cudnnConvAlgoSearch,
            arenaExtendStrategy: $this->arenaExtendStrategy,
            doCopyInDefaultStream: $this->doCopyInDefaultStream,
            tunableOpEnable: $this->tunableOpEnable,
            tunableOpTuningEnable: $this->tunableOpTuningEnable,
            tunableOpMaxTuningDurationMs: $this->tunableOpMaxTuningDurationMs,
        );
    }

    public function withGpuMemoryLimit(?int $gpuMemoryLimit): self
    {
        return new self(
            deviceId: $this->deviceId,
            gpuMemoryLimit: $gpuMemoryLimit,
            cudnnConvAlgoSearch: $this->cudnnConvAlgoSearch,
            arenaExtendStrategy: $this->arenaExtendStrategy,
            doCopyInDefaultStream: $this->doCopyInDefaultStream,
            tunableOpEnable: $this->tunableOpEnable,
            tunableOpTuningEnable: $this->tunableOpTuningEnable,
            tunableOpMaxTuningDurationMs: $this->tunableOpMaxTuningDurationMs,
        );
    }

    public function withCudnnConvAlgoSearch(?CudnnConvAlgoSearch $cudnnConvAlgoSearch): self
    {
        return new self(
            deviceId: $this->deviceId,
            gpuMemoryLimit: $this->gpuMemoryLimit,
            cudnnConvAlgoSearch: $cudnnConvAlgoSearch,
            arenaExtendStrategy: $this->arenaExtendStrategy,
            doCopyInDefaultStream: $this->doCopyInDefaultStream,
            tunableOpEnable: $this->tunableOpEnable,
            tunableOpTuningEnable: $this->tunableOpTuningEnable,
            tunableOpMaxTuningDurationMs: $this->tunableOpMaxTuningDurationMs,
        );
    }

    public function withArenaExtendStrategy(?ArenaExtendStrategy $arenaExtendStrategy): self
    {
        return new self(
            deviceId: $this->deviceId,
            gpuMemoryLimit: $this->gpuMemoryLimit,
            cudnnConvAlgoSearch: $this->cudnnConvAlgoSearch,
            arenaExtendStrategy: $arenaExtendStrategy,
            doCopyInDefaultStream: $this->doCopyInDefaultStream,
            tunableOpEnable: $this->tunableOpEnable,
            tunableOpTuningEnable: $this->tunableOpTuningEnable,
            tunableOpMaxTuningDurationMs: $this->tunableOpMaxTuningDurationMs,
        );
    }

    public function withDoCopyInDefaultStream(bool $doCopyInDefaultStream): self
    {
        return new self(
            deviceId: $this->deviceId,
            gpuMemoryLimit: $this->gpuMemoryLimit,
            cudnnConvAlgoSearch: $this->cudnnConvAlgoSearch,
            arenaExtendStrategy: $this->arenaExtendStrategy,
            doCopyInDefaultStream: $doCopyInDefaultStream,
            tunableOpEnable: $this->tunableOpEnable,
            tunableOpTuningEnable: $this->tunableOpTuningEnable,
            tunableOpMaxTuningDurationMs: $this->tunableOpMaxTuningDurationMs,
        );
    }

    public function withTunableOpEnable(bool $tunableOpEnable): self
    {
        return new self(
            deviceId: $this->deviceId,
            gpuMemoryLimit: $this->gpuMemoryLimit,
            cudnnConvAlgoSearch: $this->cudnnConvAlgoSearch,
            arenaExtendStrategy: $this->arenaExtendStrategy,
            doCopyInDefaultStream: $this->doCopyInDefaultStream,
            tunableOpEnable: $tunableOpEnable,
            tunableOpTuningEnable: $this->tunableOpTuningEnable,
            tunableOpMaxTuningDurationMs: $this->tunableOpMaxTuningDurationMs,
        );
    }

    public function withTunableOpTuningEnable(bool $tunableOpTuningEnable): self
    {
        return new self(
            deviceId: $this->deviceId,
            gpuMemoryLimit: $this->gpuMemoryLimit,
            cudnnConvAlgoSearch: $this->cudnnConvAlgoSearch,
            arenaExtendStrategy: $this->arenaExtendStrategy,
            doCopyInDefaultStream: $this->doCopyInDefaultStream,
            tunableOpEnable: $this->tunableOpEnable,
            tunableOpTuningEnable: $tunableOpTuningEnable,
            tunableOpMaxTuningDurationMs: $this->tunableOpMaxTuningDurationMs,
        );
    }

    public function withTunableOpMaxTuningDurationMs(int $tunableOpMaxTuningDurationMs): self
    {
        return new self(
            deviceId: $this->deviceId,
            gpuMemoryLimit: $this->gpuMemoryLimit,
            cudnnConvAlgoSearch: $this->cudnnConvAlgoSearch,
            arenaExtendStrategy: $this->arenaExtendStrategy,
            doCopyInDefaultStream: $this->doCopyInDefaultStream,
            tunableOpEnable: $this->tunableOpEnable,
            tunableOpTuningEnable: $this->tunableOpTuningEnable,
            tunableOpMaxTuningDurationMs: $tunableOpMaxTuningDurationMs,
        );
    }

    protected function createOptions(): CData
    {
        return Lib::api()->createCUDAProviderOptions();
    }

    protected function updateOptions(CData $options): void
    {
        $keyValues = [];

        $keyValues['device_id'] = (string) $this->deviceId;

        if (null !== $this->gpuMemoryLimit) {
            $keyValues['gpu_mem_limit'] = (string) $this->gpuMemoryLimit;
        }

        if (null !== $this->cudnnConvAlgoSearch) {
            $keyValues['cudnn_conv_algo_search'] = (string) $this->cudnnConvAlgoSearch->value;
        }

        if (null !== $this->arenaExtendStrategy) {
            $keyValues['arena_extend_strategy'] = (string) $this->arenaExtendStrategy->value;
        }

        $keyValues['do_copy_in_default_stream'] = $this->doCopyInDefaultStream ? '1' : '0';
        $keyValues['tunable_op_enable'] = $this->tunableOpEnable ? '1' : '0';
        $keyValues['tunable_op_tuning_enable'] = $this->tunableOpTuningEnable ? '1' : '0';
        $keyValues['tunable_op_max_tuning_duration_ms'] = (string) $this->tunableOpMaxTuningDurationMs;

        Lib::api()->updateCUDAProviderOptions($options, $keyValues);
    }

    protected function appendProvider(CData $sessionOptions, CData $providerOptions): void
    {
        Lib::api()->sessionOptionsAppendExecutionProviderCUDA_V2($sessionOptions, $providerOptions);
    }

    protected function releaseOptions(CData $options): void
    {
        Lib::api()->releaseCUDAProviderOptions($options);
    }
}

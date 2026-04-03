<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime;

use FFI\CData;
use PhpMlKit\ONNXRuntime\Contracts\Disposable;
use PhpMlKit\ONNXRuntime\Enums\ExecutionMode;
use PhpMlKit\ONNXRuntime\Enums\ExecutionProvider;
use PhpMlKit\ONNXRuntime\Enums\GraphOptimizationLevel;
use PhpMlKit\ONNXRuntime\Enums\LoggingLevel;
use PhpMlKit\ONNXRuntime\FFI\Lib;
use PhpMlKit\ONNXRuntime\Providers\CoreMLProviderOptions;
use PhpMlKit\ONNXRuntime\Providers\CudaProviderOptions;
use PhpMlKit\ONNXRuntime\Providers\DirectMLProviderOptions;
use PhpMlKit\ONNXRuntime\Providers\ProviderOptions;
use PhpMlKit\ONNXRuntime\Providers\TensorRTProviderOptions;

/**
 * Session configuration options for InferenceSession.
 */
final class SessionOptions implements Disposable
{
    /** @var null|CData Lazy-initialized ONNX Runtime session options handle */
    private ?CData $handle = null;

    /**
     * @param null|GraphOptimizationLevel $graphOptimizationLevel Graph optimization level
     * @param null|ExecutionMode          $executionMode          Execution mode (sequential or parallel)
     * @param null|int                    $interOpNumThreads      Number of threads for inter-op parallelism
     * @param null|int                    $intraOpNumThreads      Number of threads for intra-op parallelism
     * @param bool                        $enableCpuMemArena      Enable CPU memory arena
     * @param bool                        $enableMemPattern       Enable memory pattern optimization
     * @param null|string                 $optimizedModelFilepath Path to save the optimized model
     * @param bool                        $enableProfiling        Enable profiling
     * @param null|string                 $profileFilePrefix      Profiling file prefix
     * @param null|LoggingLevel           $logSeverityLevel       Log severity level
     * @param null|LoggingLevel           $logVerbosityLevel      Log verbosity level
     * @param array<ProviderOptions>      $providers              Execution provider configurations
     */
    public function __construct(
        public readonly ?GraphOptimizationLevel $graphOptimizationLevel = null,
        public readonly ?ExecutionMode $executionMode = null,
        public readonly ?int $interOpNumThreads = null,
        public readonly ?int $intraOpNumThreads = null,
        public readonly bool $enableCpuMemArena = true,
        public readonly bool $enableMemPattern = true,
        public readonly ?string $optimizedModelFilepath = null,
        public readonly bool $enableProfiling = false,
        public readonly ?string $profileFilePrefix = null,
        public readonly ?LoggingLevel $logSeverityLevel = null,
        public readonly ?LoggingLevel $logVerbosityLevel = null,
        public readonly array $providers = [],
    ) {
        foreach ($this->providers as $provider) {
            if (!$provider instanceof ProviderOptions) {
                throw new \InvalidArgumentException(
                    'All providers must implement ProviderOptions interface'
                );
            }
        }
    }

    public function __destruct()
    {
        $this->dispose();
    }

    /**
     * Get the ONNX Runtime session options handle.
     *
     * Lazily creates the handle on first call.
     *
     * @return CData ONNX Runtime session options handle
     */
    public function getHandle(): CData
    {
        if (null === $this->handle) {
            $this->handle = $this->createHandle();
        }

        return $this->handle;
    }

    /**
     * Release the session options handle.
     *
     * Safe to call multiple times. Called automatically on destruction.
     */
    public function dispose(): void
    {
        if (null !== $this->handle) {
            Lib::api()->releaseSessionOptions($this->handle);
            $this->handle = null;
        }
    }

    /**
     * Create default options.
     */
    public static function default(): self
    {
        return new self();
    }

    /**
     * Create options optimized for CPU inference.
     */
    public static function cpuOptimized(): self
    {
        return new self(
            graphOptimizationLevel: GraphOptimizationLevel::ENABLE_ALL,
            executionMode: ExecutionMode::SEQUENTIAL,
            enableCpuMemArena: true,
            enableMemPattern: true,
        );
    }

    /**
     * Create options optimized for parallel GPU inference.
     */
    public static function gpuParallel(): self
    {
        return new self(
            graphOptimizationLevel: GraphOptimizationLevel::ENABLE_ALL,
            executionMode: ExecutionMode::PARALLEL,
            enableCpuMemArena: true,
            enableMemPattern: true,
            providers: [new CudaProviderOptions()],
        );
    }

    /**
     * Create options for debugging/profiling.
     */
    public static function debug(): self
    {
        return new self(
            graphOptimizationLevel: GraphOptimizationLevel::DISABLE_ALL,
            enableProfiling: true,
            logSeverityLevel: LoggingLevel::VERBOSE,
        );
    }

    public function withGraphOptimizationLevel(GraphOptimizationLevel $level): self
    {
        return new self(
            graphOptimizationLevel: $level,
            executionMode: $this->executionMode,
            interOpNumThreads: $this->interOpNumThreads,
            intraOpNumThreads: $this->intraOpNumThreads,
            enableCpuMemArena: $this->enableCpuMemArena,
            enableMemPattern: $this->enableMemPattern,
            optimizedModelFilepath: $this->optimizedModelFilepath,
            enableProfiling: $this->enableProfiling,
            profileFilePrefix: $this->profileFilePrefix,
            logSeverityLevel: $this->logSeverityLevel,
            logVerbosityLevel: $this->logVerbosityLevel,
            providers: $this->providers,
        );
    }

    public function withExecutionMode(ExecutionMode $mode): self
    {
        return new self(
            graphOptimizationLevel: $this->graphOptimizationLevel,
            executionMode: $mode,
            interOpNumThreads: $this->interOpNumThreads,
            intraOpNumThreads: $this->intraOpNumThreads,
            enableCpuMemArena: $this->enableCpuMemArena,
            enableMemPattern: $this->enableMemPattern,
            optimizedModelFilepath: $this->optimizedModelFilepath,
            enableProfiling: $this->enableProfiling,
            profileFilePrefix: $this->profileFilePrefix,
            logSeverityLevel: $this->logSeverityLevel,
            logVerbosityLevel: $this->logVerbosityLevel,
            providers: $this->providers,
        );
    }

    public function withInterOpThreads(int $threads): self
    {
        return new self(
            graphOptimizationLevel: $this->graphOptimizationLevel,
            executionMode: $this->executionMode,
            interOpNumThreads: $threads,
            intraOpNumThreads: $this->intraOpNumThreads,
            enableCpuMemArena: $this->enableCpuMemArena,
            enableMemPattern: $this->enableMemPattern,
            optimizedModelFilepath: $this->optimizedModelFilepath,
            enableProfiling: $this->enableProfiling,
            profileFilePrefix: $this->profileFilePrefix,
            logSeverityLevel: $this->logSeverityLevel,
            logVerbosityLevel: $this->logVerbosityLevel,
            providers: $this->providers,
        );
    }

    public function withIntraOpThreads(int $threads): self
    {
        return new self(
            graphOptimizationLevel: $this->graphOptimizationLevel,
            executionMode: $this->executionMode,
            interOpNumThreads: $this->interOpNumThreads,
            intraOpNumThreads: $threads,
            enableCpuMemArena: $this->enableCpuMemArena,
            enableMemPattern: $this->enableMemPattern,
            optimizedModelFilepath: $this->optimizedModelFilepath,
            enableProfiling: $this->enableProfiling,
            profileFilePrefix: $this->profileFilePrefix,
            logSeverityLevel: $this->logSeverityLevel,
            logVerbosityLevel: $this->logVerbosityLevel,
            providers: $this->providers,
        );
    }

    public function withCpuMemArena(bool $enable = true): self
    {
        return new self(
            graphOptimizationLevel: $this->graphOptimizationLevel,
            executionMode: $this->executionMode,
            interOpNumThreads: $this->interOpNumThreads,
            intraOpNumThreads: $this->intraOpNumThreads,
            enableCpuMemArena: $enable,
            enableMemPattern: $this->enableMemPattern,
            optimizedModelFilepath: $this->optimizedModelFilepath,
            enableProfiling: $this->enableProfiling,
            profileFilePrefix: $this->profileFilePrefix,
            logSeverityLevel: $this->logSeverityLevel,
            logVerbosityLevel: $this->logVerbosityLevel,
            providers: $this->providers,
        );
    }

    public function withMemPattern(bool $enable = true): self
    {
        return new self(
            graphOptimizationLevel: $this->graphOptimizationLevel,
            executionMode: $this->executionMode,
            interOpNumThreads: $this->interOpNumThreads,
            intraOpNumThreads: $this->intraOpNumThreads,
            enableCpuMemArena: $this->enableCpuMemArena,
            enableMemPattern: $enable,
            optimizedModelFilepath: $this->optimizedModelFilepath,
            enableProfiling: $this->enableProfiling,
            profileFilePrefix: $this->profileFilePrefix,
            logSeverityLevel: $this->logSeverityLevel,
            logVerbosityLevel: $this->logVerbosityLevel,
            providers: $this->providers,
        );
    }

    /**
     * Set the optimized model filepath.
     *
     * @param string $filepath Path to save the optimized model
     */
    public function withOptimizedModelFilepath(string $filepath): self
    {
        return new self(
            graphOptimizationLevel: $this->graphOptimizationLevel,
            executionMode: $this->executionMode,
            interOpNumThreads: $this->interOpNumThreads,
            intraOpNumThreads: $this->intraOpNumThreads,
            enableCpuMemArena: $this->enableCpuMemArena,
            enableMemPattern: $this->enableMemPattern,
            optimizedModelFilepath: $filepath,
            enableProfiling: $this->enableProfiling,
            profileFilePrefix: $this->profileFilePrefix,
            logSeverityLevel: $this->logSeverityLevel,
            logVerbosityLevel: $this->logVerbosityLevel,
            providers: $this->providers,
        );
    }

    public function withProfiling(bool $enable = true, ?string $filePrefix = null): self
    {
        return new self(
            graphOptimizationLevel: $this->graphOptimizationLevel,
            executionMode: $this->executionMode,
            interOpNumThreads: $this->interOpNumThreads,
            intraOpNumThreads: $this->intraOpNumThreads,
            enableCpuMemArena: $this->enableCpuMemArena,
            enableMemPattern: $this->enableMemPattern,
            optimizedModelFilepath: $this->optimizedModelFilepath,
            enableProfiling: $enable,
            profileFilePrefix: $filePrefix ?? $this->profileFilePrefix,
            logSeverityLevel: $this->logSeverityLevel,
            logVerbosityLevel: $this->logVerbosityLevel,
            providers: $this->providers,
        );
    }

    public function withLogLevel(LoggingLevel $severityLevel, ?LoggingLevel $verbosityLevel = null): self
    {
        return new self(
            graphOptimizationLevel: $this->graphOptimizationLevel,
            executionMode: $this->executionMode,
            interOpNumThreads: $this->interOpNumThreads,
            intraOpNumThreads: $this->intraOpNumThreads,
            enableCpuMemArena: $this->enableCpuMemArena,
            enableMemPattern: $this->enableMemPattern,
            optimizedModelFilepath: $this->optimizedModelFilepath,
            enableProfiling: $this->enableProfiling,
            profileFilePrefix: $this->profileFilePrefix,
            logSeverityLevel: $severityLevel,
            logVerbosityLevel: $verbosityLevel ?? $this->logVerbosityLevel,
            providers: $this->providers,
        );
    }

    /**
     * Add a provider to the session options.
     *
     * Providers are appended in the order they are added.
     *
     * @param ProviderOptions $provider Provider configuration
     */
    public function withProvider(ProviderOptions $provider): self
    {
        $providers = $this->providers;
        $providers[] = $provider;

        return new self(
            graphOptimizationLevel: $this->graphOptimizationLevel,
            executionMode: $this->executionMode,
            interOpNumThreads: $this->interOpNumThreads,
            intraOpNumThreads: $this->intraOpNumThreads,
            enableCpuMemArena: $this->enableCpuMemArena,
            enableMemPattern: $this->enableMemPattern,
            optimizedModelFilepath: $this->optimizedModelFilepath,
            enableProfiling: $this->enableProfiling,
            profileFilePrefix: $this->profileFilePrefix,
            logSeverityLevel: $this->logSeverityLevel,
            logVerbosityLevel: $this->logVerbosityLevel,
            providers: $providers,
        );
    }

    /**
     * Add CUDA provider.
     *
     * @param null|CudaProviderOptions $options CUDA provider options (uses defaults if null)
     */
    public function withCudaProvider(?CudaProviderOptions $options = null): self
    {
        return $this->withProvider($options ?? CudaProviderOptions::default());
    }

    /**
     * Add CoreML provider.
     *
     * @param null|CoreMLProviderOptions $options CoreML provider options (uses defaults if null)
     */
    public function withCoreMLProvider(?CoreMLProviderOptions $options = null): self
    {
        return $this->withProvider($options ?? CoreMLProviderOptions::default());
    }

    /**
     * Add DirectML provider (Windows only).
     *
     * @param null|DirectMLProviderOptions $options DirectML provider options (uses defaults if null)
     */
    public function withDirectMLProvider(?DirectMLProviderOptions $options = null): self
    {
        return $this->withProvider($options ?? DirectMLProviderOptions::default());
    }

    /**
     * Add TensorRT provider.
     *
     * @param null|TensorRTProviderOptions $options TensorRT provider options (uses defaults if null)
     */
    public function withTensorRTProvider(?TensorRTProviderOptions $options = null): self
    {
        return $this->withProvider($options ?? TensorRTProviderOptions::default());
    }

    /**
     * Remove a provider from the session options.
     *
     * @param ExecutionProvider $provider Provider type to remove
     */
    public function withoutProvider(ExecutionProvider $provider): self
    {
        $providers = array_values(array_filter(
            $this->providers,
            static fn (ProviderOptions $p) => $p->getProvider() !== $provider
        ));

        return new self(
            graphOptimizationLevel: $this->graphOptimizationLevel,
            executionMode: $this->executionMode,
            interOpNumThreads: $this->interOpNumThreads,
            intraOpNumThreads: $this->intraOpNumThreads,
            enableCpuMemArena: $this->enableCpuMemArena,
            enableMemPattern: $this->enableMemPattern,
            optimizedModelFilepath: $this->optimizedModelFilepath,
            enableProfiling: $this->enableProfiling,
            profileFilePrefix: $this->profileFilePrefix,
            logSeverityLevel: $this->logSeverityLevel,
            logVerbosityLevel: $this->logVerbosityLevel,
            providers: $providers,
        );
    }

    /**
     * Create the ONNX Runtime session options handle.
     *
     * Applies all configured options to the handle.
     *
     * @return CData New session options handle
     */
    private function createHandle(): CData
    {
        $api = Lib::api();

        $handle = $api->createSessionOptions();

        if (null !== $this->graphOptimizationLevel) {
            $api->setSessionGraphOptimizationLevel($handle, $this->graphOptimizationLevel->value);
        }

        if (null !== $this->executionMode) {
            $api->setSessionExecutionMode($handle, $this->executionMode->value);
        }

        if (null !== $this->interOpNumThreads) {
            $api->setInterOpNumThreads($handle, $this->interOpNumThreads);
        }

        if (null !== $this->intraOpNumThreads) {
            $api->setIntraOpNumThreads($handle, $this->intraOpNumThreads);
        }

        if ($this->enableCpuMemArena) {
            $api->enableCpuMemArena($handle);
        } else {
            $api->disableCpuMemArena($handle);
        }

        if ($this->enableMemPattern) {
            $api->enableMemPattern($handle);
        } else {
            $api->disableMemPattern($handle);
        }

        if (null !== $this->optimizedModelFilepath) {
            $api->setOptimizedModelFilepath($handle, $this->optimizedModelFilepath);
        }

        if ($this->enableProfiling) {
            $prefix = $this->profileFilePrefix ?? 'onnxruntime_profile_';
            $api->enableProfiling($handle, $prefix);
        }

        if (null !== $this->logSeverityLevel) {
            $api->setSessionLogSeverityLevel($handle, $this->logSeverityLevel);
        }

        if (null !== $this->logVerbosityLevel) {
            $api->setSessionLogVerbosityLevel($handle, $this->logVerbosityLevel);
        }

        foreach ($this->providers as $provider) {
            $provider->applyToSession($handle);
        }

        return $handle;
    }
}

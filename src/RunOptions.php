<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime;

use FFI\CData;
use PhpMlKit\ONNXRuntime\Contracts\Disposable;
use PhpMlKit\ONNXRuntime\Enums\LoggingLevel;
use PhpMlKit\ONNXRuntime\FFI\Lib;

/**
 * Run configuration options for inference.
 *
 * Immutable value object with fluent methods for configuration.
 * Manages its own ONNX Runtime run options handle via RAII.
 *
 * Run options allow per-inference configuration that overrides session defaults,
 * such as custom logging levels, run tags for profiling, and termination control.
 */
final class RunOptions implements Disposable
{
    /** @var null|CData Lazy-initialized ONNX Runtime run options handle */
    private ?CData $handle = null;

    /**
     * @param null|LoggingLevel $logVerbosityLevel Log verbosity level
     * @param null|LoggingLevel $logSeverityLevel  Log severity level
     * @param null|string       $runTag            Run tag for profiling/identification
     * @param bool              $terminate         Whether to request termination
     */
    public function __construct(
        public readonly ?LoggingLevel $logVerbosityLevel = null,
        public readonly ?LoggingLevel $logSeverityLevel = null,
        public readonly ?string $runTag = null,
        public readonly bool $terminate = false,
    ) {}

    public function __destruct()
    {
        $this->dispose();
    }

    /**
     * Get the ONNX Runtime run options handle.
     *
     * Lazily creates the handle on first call.
     *
     * @return CData ONNX Runtime run options handle
     */
    public function getHandle(): CData
    {
        if (null === $this->handle) {
            $this->handle = $this->createHandle();
        }

        return $this->handle;
    }

    /**
     * Release the run options handle.
     *
     * Safe to call multiple times. Called automatically on destruction.
     */
    public function dispose(): void
    {
        if (null !== $this->handle) {
            Lib::api()->releaseRunOptions($this->handle);
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
     * Create options for debugging with verbose logging.
     */
    public static function debug(): self
    {
        return new self(
            logSeverityLevel: LoggingLevel::VERBOSE,
        );
    }

    /**
     * Create options with a run tag for profiling.
     *
     * @param string $tag Run tag for profiling/identification
     */
    public static function withTag(string $tag): self
    {
        return new self(runTag: $tag);
    }

    /**
     * Create options for silent operation (fatal errors only).
     */
    public static function silent(): self
    {
        return new self(
            logSeverityLevel: LoggingLevel::FATAL,
        );
    }

    public function withLogVerbosityLevel(LoggingLevel $level): self
    {
        return new self(
            logVerbosityLevel: $level,
            logSeverityLevel: $this->logSeverityLevel,
            runTag: $this->runTag,
            terminate: $this->terminate,
        );
    }

    public function withLogSeverityLevel(LoggingLevel $level): self
    {
        return new self(
            logVerbosityLevel: $this->logVerbosityLevel,
            logSeverityLevel: $level,
            runTag: $this->runTag,
            terminate: $this->terminate,
        );
    }

    public function withRunTag(string $tag): self
    {
        return new self(
            logVerbosityLevel: $this->logVerbosityLevel,
            logSeverityLevel: $this->logSeverityLevel,
            runTag: $tag,
            terminate: $this->terminate,
        );
    }

    public function withTerminate(bool $terminate = true): self
    {
        return new self(
            logVerbosityLevel: $this->logVerbosityLevel,
            logSeverityLevel: $this->logSeverityLevel,
            runTag: $this->runTag,
            terminate: $terminate,
        );
    }

    /**
     * Create the ONNX Runtime run options handle.
     *
     * Applies all configured options to the handle.
     *
     * @return CData New run options handle
     */
    private function createHandle(): CData
    {
        $api = Lib::api();

        $handle = $api->createRunOptions();

        if (null !== $this->logVerbosityLevel) {
            $api->setRunOptionsLogVerbosityLevel($handle, $this->logVerbosityLevel);
        }

        if (null !== $this->logSeverityLevel) {
            $api->setRunOptionsLogSeverityLevel($handle, $this->logSeverityLevel);
        }

        if (null !== $this->runTag) {
            $api->setRunOptionsRunTag($handle, $this->runTag);
        }

        if ($this->terminate) {
            $api->setRunOptionsTerminate($handle, true);
        }

        return $handle;
    }
}

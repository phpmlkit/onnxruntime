<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime;

use FFI\CData;
use PhpMlKit\ONNXRuntime\Contracts\Disposable;
use PhpMlKit\ONNXRuntime\Enums\LoggingLevel;
use PhpMlKit\ONNXRuntime\Exceptions\InvalidOperationException;
use PhpMlKit\ONNXRuntime\FFI\Lib;
use PhpMlKit\ONNXRuntime\FFI\Mutex;

/**
 * ONNX Runtime Environment.
 *
 * Manages the global ONNX Runtime environment with reference counting.
 * This is an internal class - users should not interact with it directly.
 * The environment is automatically managed by InferenceSession instances.
 *
 * The environment is a singleton resource that is reference-counted.
 * When the last holder releases it, the environment is automatically freed.
 *
 * @internal this class is not part of the public API and may change without notice
 */
final class Environment implements Disposable
{
    private static ?self $instance = null;

    private CData $handle;
    private int $refCount = 0;
    private Mutex $mutex;
    private bool $disposed = false;

    private function __construct()
    {
        $this->mutex = new Mutex();

        $api = Lib::api();

        $this->handle = $api->createEnv(LoggingLevel::FATAL, 'onnxruntime');

        // https://github.com/microsoft/onnxruntime/blob/main/docs/Privacy.md
        $api->disableTelemetryEvents($this->handle);
    }

    /**
     * Acquire an environment instance.
     *
     * This method is thread-safe. If an environment doesn't exist, it will be created.
     * The returned instance must be disposed when no longer needed.
     */
    public static function instance(): self
    {
        if (null === self::$instance) {
            self::$instance = new self();
        }

        $instance = self::$instance;

        $instance->mutex->lock();

        try {
            ++$instance->refCount;

            return $instance;
        } finally {
            $instance->mutex->unlock();
        }
    }

    /**
     * Get the native environment handle.
     *
     * @return CData The ONNX Runtime environment handle
     *
     * @throws InvalidOperationException If the environment has been disposed
     */
    public function getHandle(): CData
    {
        if ($this->disposed) {
            throw new InvalidOperationException('Environment has been disposed');
        }

        return $this->handle;
    }

    /**
     * Release this environment reference.
     *
     * Safe to call multiple times. When the last reference is released,
     * the underlying ONNX environment is freed.
     */
    public function dispose(): void
    {
        if ($this->disposed) {
            return;
        }

        $this->mutex->lock();

        try {
            --$this->refCount;

            if (0 === $this->refCount) {
                $api = Lib::api();
                $api->releaseEnv($this->handle);

                $this->mutex->dispose();

                self::$instance = null;
                $this->disposed = true;
            }
        } finally {
            if (isset($this->mutex) && $this->mutex->isLocked()) {
                $this->mutex->unlock();
            }
        }
    }
}

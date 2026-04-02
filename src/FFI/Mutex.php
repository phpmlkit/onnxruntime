<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\FFI;

use FFI\CData;
use PhpMlKit\ONNXRuntime\Contracts\Disposable;

/**
 * Platform-agnostic mutex wrapper for thread synchronization.
 *
 * Uses pthread mutexes on POSIX systems and Critical Sections on Windows.
 * Automatically tracks lock state and provides safe cleanup.
 *
 * @internal this class is not part of the public API and may change without notice
 */
final class Mutex implements Disposable
{
    private CData $handle;
    private bool $locked = false;
    private bool $disposed = false;

    /** @var null|\FFI FFI instance for libc synchronization primitives */
    private static $libc;

    public function __construct()
    {
        $this->handle = self::createHandle();
    }

    public function __destruct()
    {
        $this->dispose();
    }

    /**
     * Acquire the lock.
     *
     * Blocks until the lock is acquired.
     *
     * @throws \RuntimeException If the lock cannot be acquired
     * @throws \RuntimeException If the mutex has been disposed
     */
    public function lock(): void
    {
        if ($this->disposed) {
            throw new \RuntimeException('Cannot lock disposed mutex');
        }

        if ($this->locked) {
            throw new \RuntimeException('Mutex is already locked by this instance');
        }

        $libc = self::getLibc();

        if (\PHP_OS_FAMILY !== 'Windows') {
            $result = $libc->pthread_mutex_lock(\FFI::addr($this->handle));
            if (0 !== $result) {
                throw new \RuntimeException("Failed to acquire lock: {$result}");
            }
        } else {
            $libc->EnterCriticalSection(\FFI::addr($this->handle));
        }

        $this->locked = true;
    }

    /**
     * Release the lock.
     *
     * Safe to call even if not locked (no-op).
     *
     * @throws \RuntimeException If the mutex has been disposed
     */
    public function unlock(): void
    {
        if ($this->disposed) {
            throw new \RuntimeException('Cannot unlock disposed mutex');
        }

        if (!$this->locked) {
            return;
        }

        $libc = self::getLibc();

        if (\PHP_OS_FAMILY !== 'Windows') {
            $libc->pthread_mutex_unlock(\FFI::addr($this->handle));
        } else {
            $libc->LeaveCriticalSection(\FFI::addr($this->handle));
        }

        $this->locked = false;
    }

    /**
     * Destroy the mutex and release resources.
     *
     * If the mutex is currently locked, it will be unlocked first.
     * Safe to call multiple times.
     */
    public function dispose(): void
    {
        if ($this->disposed) {
            return;
        }

        // Unlock if still locked to avoid undefined behavior
        if ($this->locked) {
            $this->unlock();
        }

        $libc = self::getLibc();

        if (\PHP_OS_FAMILY !== 'Windows') {
            $libc->pthread_mutex_destroy(\FFI::addr($this->handle));
        } else {
            $libc->DeleteCriticalSection(\FFI::addr($this->handle));
        }

        $this->disposed = true;
    }

    /**
     * Check if the mutex is currently locked.
     */
    public function isLocked(): bool
    {
        return $this->locked;
    }

    /**
     * Create the native mutex handle.
     */
    private static function createHandle(): CData
    {
        $libc = self::getLibc();

        if (\PHP_OS_FAMILY !== 'Windows') {
            $handle = $libc->new('pthread_mutex_t');
            $libc->pthread_mutex_init(\FFI::addr($handle), null);

            return $handle;
        }

        $handle = $libc->new('CRITICAL_SECTION');
        $libc->InitializeCriticalSection(\FFI::addr($handle));

        return $handle;
    }

    /**
     * Get the libc FFI instance for synchronization primitives.
     */
    private static function getLibc(): \FFI
    {
        if (null === self::$libc) {
            if (\PHP_OS_FAMILY === 'Darwin') {
                // macOS uses libSystem which includes pthread
                self::$libc = \FFI::cdef('
                    typedef struct pthread_mutex_t {
                        unsigned long __sig;
                        unsigned char __opaque[56];
                    } pthread_mutex_t;
                    int pthread_mutex_init(pthread_mutex_t *mutex, const void *attr);
                    int pthread_mutex_lock(pthread_mutex_t *mutex);
                    int pthread_mutex_unlock(pthread_mutex_t *mutex);
                    int pthread_mutex_destroy(pthread_mutex_t *mutex);
                ', 'libSystem.dylib');
            } elseif (\PHP_OS_FAMILY !== 'Windows') {
                // Linux uses libc.so.6
                self::$libc = \FFI::cdef('
                    typedef struct pthread_mutex_t {
                        unsigned char __opaque[40];
                    } pthread_mutex_t;
                    int pthread_mutex_init(pthread_mutex_t *mutex, const void *attr);
                    int pthread_mutex_lock(pthread_mutex_t *mutex);
                    int pthread_mutex_unlock(pthread_mutex_t *mutex);
                    int pthread_mutex_destroy(pthread_mutex_t *mutex);
                ', 'libc.so.6');
            } else {
                // Windows uses Critical Sections from kernel32
                self::$libc = \FFI::cdef('
                    typedef struct _RTL_CRITICAL_SECTION {
                        void *DebugInfo;
                        long LockCount;
                        long RecursionCount;
                        void *OwningThread;
                        void *LockSemaphore;
                        unsigned long SpinCount;
                    } CRITICAL_SECTION, *PCRITICAL_SECTION, *LPCRITICAL_SECTION;
                    void InitializeCriticalSection(CRITICAL_SECTION *lpCriticalSection);
                    void EnterCriticalSection(CRITICAL_SECTION *lpCriticalSection);
                    void LeaveCriticalSection(CRITICAL_SECTION *lpCriticalSection);
                    void DeleteCriticalSection(CRITICAL_SECTION *lpCriticalSection);
                ', 'kernel32.dll');
            }
        }

        return self::$libc;
    }
}

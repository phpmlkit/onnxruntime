<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Contracts;

/**
 * Interface for resources that require explicit cleanup.
 *
 * Classes implementing this interface manage native or external resources
 * that should be released when no longer needed. While resources are
 * automatically cleaned up when the object goes out of scope (via destructor),
 * explicit disposal allows for deterministic resource management.
 *
 * The dispose() method is safe to call multiple times - subsequent calls
 * have no effect.
 *
 * @example
 * $session = InferenceSession::fromFile('model.onnx');
 * try {
 *     $outputs = $session->run($inputs);
 * } finally {
 *     $session->dispose(); // Explicit cleanup
 * }
 */
interface Disposable
{
    /**
     * Release all resources held by this object.
     *
     * This method releases native handles, memory, or other external
     * resources. It is safe to call multiple times - subsequent calls
     * have no effect.
     *
     * After disposal, the object should not be used except for checking
     * if it has been disposed.
     */
    public function dispose(): void;
}

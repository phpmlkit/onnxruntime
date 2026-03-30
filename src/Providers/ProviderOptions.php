<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Providers;

use FFI\CData;
use PhpMlKit\ONNXRuntime\Enums\ExecutionProvider;

/**
 * Interface for execution provider configuration options.
 *
 * All provider options must implement this interface to be used
 * with SessionOptions. Each provider handles its own initialization
 * and appending to the session options.
 */
interface ProviderOptions
{
    /**
     * Get the execution provider type.
     */
    public function getProvider(): ExecutionProvider;

    /**
     * Apply this provider configuration to a session options handle.
     *
     * This method is responsible for:
     * 1. Creating any necessary provider-specific options structs
     * 2. Configuring the provider with the specified options
     * 3. Appending the provider to the session options
     * 4. Cleaning up any temporary resources
     *
     * @param CData $sessionOptions The ONNX Runtime session options handle
     */
    public function applyToSession(CData $sessionOptions): void;

    /**
     * Get the provider options as an array for serialization.
     *
     * @return array<string, mixed>
     */
    public function toArray(): array;
}

<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Providers;

use FFI\CData;

/**
 * Abstract base class for provider options using the V2 API.
 *
 * V2 providers (like CUDA and TensorRT) use an opaque options struct
 * that must be created, configured, appended, and then released.
 *
 * @template T of object
 */
abstract class V2ProviderOptions implements ProviderOptions
{
    /**
     * Create the provider options struct.
     *
     * @return CData The provider options struct handle
     */
    abstract protected function createOptions(): CData;

    /**
     * Update the provider options with configured values.
     *
     * @param CData $options The provider options struct handle
     */
    abstract protected function updateOptions(CData $options): void;

    /**
     * Append the provider to session options.
     *
     * @param CData $sessionOptions The session options handle
     * @param CData $providerOptions The provider options struct handle
     */
    abstract protected function appendProvider(CData $sessionOptions, CData $providerOptions): void;

    /**
     * Release the provider options struct.
     *
     * @param CData $options The provider options struct handle
     */
    abstract protected function releaseOptions(CData $options): void;

    public function applyToSession(CData $sessionOptions): void
    {
        $options = null;

        try {
            $options = $this->createOptions();
            $this->updateOptions($options);
            $this->appendProvider($sessionOptions, $options);
        } finally {
            if (null !== $options) {
                $this->releaseOptions($options);
            }
        }
    }
}

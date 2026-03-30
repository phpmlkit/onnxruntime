<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Providers;

use FFI\CData;
use PhpMlKit\ONNXRuntime\Enums\ExecutionProvider;
use PhpMlKit\ONNXRuntime\FFI\Lib;

/**
 * Generic execution provider options for providers using the generic API.
 *
 * Used for providers that don't have dedicated option classes yet.
 *
 * @example
 * // XNNPACK provider
 * $options = new GenericProviderOptions(
 *     provider: ExecutionProvider::XNNPACK,
 *     options: ['intra_op_num_threads' => '4'],
 * );
 *
 * // WebNN provider
 * $options = new GenericProviderOptions(
 *     provider: ExecutionProvider::WEBNN,
 * );
 */
final class GenericProviderOptions implements ProviderOptions
{
    /**
     * @param ExecutionProvider $provider The execution provider type
     * @param array<string, string> $options Provider-specific key-value options
     */
    public function __construct(
        public readonly ExecutionProvider $provider,
        public readonly array $options = [],
    ) {}

    public function getProvider(): ExecutionProvider
    {
        return $this->provider;
    }

    public function applyToSession(CData $sessionOptions): void
    {
        $providerName = match ($this->provider) {
            ExecutionProvider::QNN => 'QNN',
            ExecutionProvider::XNNPACK => 'XNNPACK',
            ExecutionProvider::WEBNN => 'WebNN',
            ExecutionProvider::WEBGPU => 'WebGPU',
            ExecutionProvider::AZURE => 'Azure',
            ExecutionProvider::VITIS_AI => 'VitisAI',
            ExecutionProvider::JS => 'JS',
            default => throw new \InvalidArgumentException(
                "Provider {$this->provider->value} should use dedicated options class"
            ),
        };

        Lib::api()->sessionOptionsAppendExecutionProvider(
            $sessionOptions,
            $providerName,
            $this->options
        );
    }

    public function toArray(): array
    {
        return [
            'provider' => $this->provider->value,
            'options' => $this->options,
        ];
    }

    /**
     * Create XNNPACK provider options.
     */
    public static function xnnpack(int $intraOpNumThreads = 0): self
    {
        return new self(
            provider: ExecutionProvider::XNNPACK,
            options: $intraOpNumThreads > 0
                ? ['intra_op_num_threads' => (string) $intraOpNumThreads]
                : [],
        );
    }

    /**
     * Create WebNN provider options.
     */
    public static function webnn(): self
    {
        return new self(provider: ExecutionProvider::WEBNN);
    }

    /**
     * Create WebGPU provider options.
     */
    public static function webgpu(): self
    {
        return new self(provider: ExecutionProvider::WEBGPU);
    }
}

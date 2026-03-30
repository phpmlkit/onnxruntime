<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Providers;

use FFI\CData;
use PhpMlKit\ONNXRuntime\Enums\DirectMLDeviceFilter;
use PhpMlKit\ONNXRuntime\Enums\DirectMLPerformancePreference;
use PhpMlKit\ONNXRuntime\Enums\ExecutionProvider;
use PhpMlKit\ONNXRuntime\FFI\Lib;

/**
 * DirectML execution provider options for Windows.
 */
final class DirectMLProviderOptions implements ProviderOptions
{
    /**
     * @param DirectMLPerformancePreference $performancePreference Performance preference (Default, HighPerformance, MinimumPower)
     * @param DirectMLDeviceFilter          $deviceFilter          Device filter (GPU, or future: NPU)
     */
    public function __construct(
        public readonly DirectMLPerformancePreference $performancePreference = DirectMLPerformancePreference::DEFAULT,
        public readonly DirectMLDeviceFilter $deviceFilter = DirectMLDeviceFilter::GPU,
    ) {}

    public function getProvider(): ExecutionProvider
    {
        return ExecutionProvider::DIRECTML;
    }

    public function applyToSession(CData $sessionOptions): void
    {
        Lib::api()->sessionOptionsAppendExecutionProviderDML2(
            $sessionOptions,
            $this->performancePreference->value,
            $this->deviceFilter->value
        );
    }

    public function toArray(): array
    {
        return [
            'performance_preference' => $this->performancePreference->value,
            'device_filter' => $this->deviceFilter->value,
        ];
    }

    /**
     * Create with default options.
     */
    public static function default(): self
    {
        return new self();
    }

    /**
     * Set performance preference.
     */
    public function withPerformancePreference(DirectMLPerformancePreference $preference): self
    {
        return new self(
            performancePreference: $preference,
            deviceFilter: $this->deviceFilter,
        );
    }

    /**
     * Set device filter.
     */
    public function withDeviceFilter(DirectMLDeviceFilter $filter): self
    {
        return new self(
            performancePreference: $this->performancePreference,
            deviceFilter: $filter,
        );
    }

    /**
     * Use high performance mode.
     */
    public function withHighPerformance(): self
    {
        return $this->withPerformancePreference(DirectMLPerformancePreference::HIGH_PERFORMANCE);
    }

    /**
     * Use minimum power mode.
     */
    public function withMinimumPower(): self
    {
        return $this->withPerformancePreference(DirectMLPerformancePreference::MINIMUM_POWER);
    }
}

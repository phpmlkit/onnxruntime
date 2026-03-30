<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Providers;

use FFI\CData;
use PhpMlKit\ONNXRuntime\Enums\CoreMLComputeUnits;
use PhpMlKit\ONNXRuntime\Enums\CoreMLModelFormat;
use PhpMlKit\ONNXRuntime\Enums\ExecutionProvider;
use PhpMlKit\ONNXRuntime\FFI\Lib;

/**
 * CoreML execution provider options for Apple devices.
 *
 * This class provides a type-safe interface to CoreML EP options as documented at:
 * @see https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html#configuration-options
 * @see https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/coreml/coreml_provider_factory.h
 */
final class CoreMLProviderOptions implements ProviderOptions
{
    private const FLAG_USE_NONE = 0x000;
    private const FLAG_USE_CPU_ONLY = 0x001;
    private const FLAG_ENABLE_ON_SUBGRAPH = 0x002;
    private const FLAG_ONLY_ENABLE_DEVICE_WITH_ANE = 0x004;
    private const FLAG_ONLY_ALLOW_STATIC_INPUT_SHAPES = 0x008;
    private const FLAG_CREATE_MLPROGRAM = 0x010;
    private const FLAG_USE_CPU_AND_GPU = 0x020;

    /**
     * @param null|CoreMLComputeUnits $computeUnits             Which compute units to use (CPU, GPU, Neural Engine)
     * @param null|CoreMLModelFormat  $modelFormat              Model format: NeuralNetwork or MLProgram
     * @param bool                    $requireStaticInputShapes Only allow static input shapes (may improve performance)
     * @param bool                    $enableOnSubgraphs        Enable CoreML on subgraphs
     * @param bool                    $specializeForANE         Only enable for devices with Apple Neural Engine
     */
    public function __construct(
        public readonly ?CoreMLComputeUnits $computeUnits = null,
        public readonly ?CoreMLModelFormat $modelFormat = null,
        public readonly bool $requireStaticInputShapes = false,
        public readonly bool $enableOnSubgraphs = true,
        public readonly bool $specializeForANE = false,
    ) {}

    public function getProvider(): ExecutionProvider
    {
        return ExecutionProvider::COREML;
    }

    public function applyToSession(CData $sessionOptions): void
    {
        $flags = $this->calculateFlags();
        Lib::api()->sessionOptionsAppendExecutionProviderCoreML($sessionOptions, $flags);
    }

    public function toArray(): array
    {
        return [
            'compute_units' => $this->computeUnits?->value,
            'model_format' => $this->modelFormat?->value,
            'require_static_input_shapes' => $this->requireStaticInputShapes,
            'enable_on_subgraphs' => $this->enableOnSubgraphs,
            'specialize_for_ane' => $this->specializeForANE,
        ];
    }

    /**
     * Create with default options (flags=0).
     *
     * Uses automatic compute unit selection and default behavior.
     */
    public static function default(): self
    {
        return new self();
    }

    /**
     * Set compute units to use.
     *
     * @param CoreMLComputeUnits $units Compute units (CPU, GPU, Neural Engine, All)
     */
    public function withComputeUnits(CoreMLComputeUnits $units): self
    {
        return new self(
            computeUnits: $units,
            modelFormat: $this->modelFormat,
            requireStaticInputShapes: $this->requireStaticInputShapes,
            enableOnSubgraphs: $this->enableOnSubgraphs,
            specializeForANE: $this->specializeForANE,
        );
    }

    /**
     * Set model format.
     *
     * @param CoreMLModelFormat $format Model format (NeuralNetwork or MLProgram)
     */
    public function withModelFormat(CoreMLModelFormat $format): self
    {
        return new self(
            computeUnits: $this->computeUnits,
            modelFormat: $format,
            requireStaticInputShapes: $this->requireStaticInputShapes,
            enableOnSubgraphs: $this->enableOnSubgraphs,
            specializeForANE: $this->specializeForANE,
        );
    }

    /**
     * Enable or disable static input shapes requirement.
     *
     * May improve performance by requiring all inputs to have static shapes.
     *
     * @param bool $enable Whether to require static input shapes
     */
    public function withStaticShapes(bool $enable = true): self
    {
        return new self(
            computeUnits: $this->computeUnits,
            modelFormat: $this->modelFormat,
            requireStaticInputShapes: $enable,
            enableOnSubgraphs: $this->enableOnSubgraphs,
            specializeForANE: $this->specializeForANE,
        );
    }

    /**
     * Enable or disable CoreML on subgraphs.
     *
     * @param bool $enable Whether to enable CoreML on subgraphs
     */
    public function withSubgraphs(bool $enable = true): self
    {
        return new self(
            computeUnits: $this->computeUnits,
            modelFormat: $this->modelFormat,
            requireStaticInputShapes: $this->requireStaticInputShapes,
            enableOnSubgraphs: $enable,
            specializeForANE: $this->specializeForANE,
        );
    }

    /**
     * Enable or disable specialization for Apple Neural Engine.
     *
     * @param bool $enable Whether to specialize for ANE
     */
    public function withANE(bool $enable = true): self
    {
        return new self(
            computeUnits: $this->computeUnits,
            modelFormat: $this->modelFormat,
            requireStaticInputShapes: $this->requireStaticInputShapes,
            enableOnSubgraphs: $this->enableOnSubgraphs,
            specializeForANE: $enable,
        );
    }

    /**
     * Calculate the bitwise flags for the CoreML legacy API.
     */
    private function calculateFlags(): int
    {
        $flags = self::FLAG_USE_NONE;

        // See: https://developer.apple.com/documentation/coreml/mlcomputeunits
        if (null !== $this->computeUnits) {
            $flags |= match ($this->computeUnits) {
                CoreMLComputeUnits::CPU_ONLY => self::FLAG_USE_CPU_ONLY,
                CoreMLComputeUnits::CPU_AND_GPU => self::FLAG_USE_CPU_AND_GPU,
                CoreMLComputeUnits::CPU_AND_NEURAL_ENGINE, CoreMLComputeUnits::ALL => 0, // These are default behavior
            };
        }

        if ($this->enableOnSubgraphs) {
            $flags |= self::FLAG_ENABLE_ON_SUBGRAPH;
        }

        if ($this->specializeForANE) {
            $flags |= self::FLAG_ONLY_ENABLE_DEVICE_WITH_ANE;
        }

        if ($this->requireStaticInputShapes) {
            $flags |= self::FLAG_ONLY_ALLOW_STATIC_INPUT_SHAPES;
        }

        // See: https://developer.apple.com/documentation/coreml/mlmodel
        if (null !== $this->modelFormat && CoreMLModelFormat::ML_PROGRAM === $this->modelFormat) {
            $flags |= self::FLAG_CREATE_MLPROGRAM;
        }

        return $flags;
    }
}

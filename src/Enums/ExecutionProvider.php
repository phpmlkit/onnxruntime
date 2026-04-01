<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Enums;

use PhpMlKit\ONNXRuntime\FFI\Lib;

/**
 * ONNX Runtime execution providers.
 *
 * Execution providers enable ONNX Runtime to run models on different hardware
 * accelerators like GPUs, NPUs, and specialized AI chips.
 */
enum ExecutionProvider: string
{
    /**
     * CPU execution provider.
     * Always available, works on all platforms.
     */
    case CPU = 'CPUExecutionProvider';

    /**
     * CUDA execution provider for NVIDIA GPUs.
     * Requires CUDA-capable GPU and CUDA runtime.
     */
    case CUDA = 'CUDAExecutionProvider';

    /**
     * ROCm execution provider for AMD GPUs.
     * Requires ROCm-capable AMD GPU.
     */
    case ROCM = 'ROCMExecutionProvider';

    /**
     * TensorRT execution provider for NVIDIA GPUs.
     * Optimized inference using NVIDIA TensorRT.
     */
    case TENSORRT = 'TensorRTExecutionProvider';

    /**
     * CoreML execution provider for Apple devices.
     * Uses Apple Neural Engine on compatible devices.
     * macOS and iOS only.
     */
    case COREML = 'CoreMLExecutionProvider';

    /**
     * DirectML execution provider for Windows.
     * Uses DirectML API for GPU acceleration on Windows.
     * Windows only.
     */
    case DIRECTML = 'DmlExecutionProvider';

    /**
     * OpenVINO execution provider.
     * Intel optimized inference for Intel CPUs, GPUs, and VPUs.
     */
    case OPENVINO = 'OpenVINOExecutionProvider';

    /**
     * QNN (Qualcomm Neural Network) execution provider.
     * For Qualcomm Snapdragon devices with NPU.
     */
    case QNN = 'QNNExecutionProvider';

    /**
     * XNNPACK execution provider.
     * Optimized for ARM and x86 mobile/embedded devices.
     */
    case XNNPACK = 'XnnpackExecutionProvider';

    /**
     * oneDNN (formerly DNNL) execution provider.
     * Intel optimized CPU inference.
     */
    case DNNL = 'DnnlExecutionProvider';

    /**
     * MIGraphX execution provider for AMD GPUs.
     * AMD's graph optimization and inference engine.
     */
    case MIGRAPHX = 'MIGraphXExecutionProvider';

    /**
     * WebNN execution provider.
     * For web browsers supporting WebNN API.
     */
    case WEBNN = 'WebNNExecutionProvider';

    /**
     * WebGPU execution provider.
     * For web browsers supporting WebGPU.
     */
    case WEBGPU = 'WebGpuExecutionProvider';

    /**
     * Azure execution provider.
     * For Azure Machine Learning endpoints.
     */
    case AZURE = 'AzureExecutionProvider';

    /**
     * Vitis AI execution provider.
     * For Xilinx/AMD adaptive SoCs and Alveo cards.
     */
    case VITIS_AI = 'VitisAIExecutionProvider';

    /**
     * JS execution provider.
     * For JavaScript/WASM environments.
     */
    case JS = 'JsExecutionProvider';

    /**
     * Get the human-readable name for this provider.
     */
    public function getDisplayName(): string
    {
        return match ($this) {
            self::CPU => 'CPU',
            self::CUDA => 'CUDA',
            self::ROCM => 'ROCm',
            self::TENSORRT => 'TensorRT',
            self::COREML => 'CoreML',
            self::DIRECTML => 'DirectML',
            self::OPENVINO => 'OpenVINO',
            self::QNN => 'QNN',
            self::XNNPACK => 'XNNPACK',
            self::DNNL => 'oneDNN',
            self::MIGRAPHX => 'MIGraphX',
            self::WEBNN => 'WebNN',
            self::WEBGPU => 'WebGPU',
            self::AZURE => 'Azure',
            self::VITIS_AI => 'Vitis AI',
            self::JS => 'JavaScript',
        };
    }

    /**
     * Check if this provider is available on the current system.
     *
     * Uses getAvailableProviders() to check what's actually compiled
     * into the ONNX Runtime library.
     */
    public function isAvailable(): bool
    {
        try {
            $availableProviders = Lib::api()->getAvailableProviders();

            return \in_array($this->value, $availableProviders, true);
        } catch (\Throwable $e) {
            return false;
        }
    }

    /**
     * Get the API type for appending this provider.
     *
     * Determines which C API function to use:
     * - LEGACY: Simple device_id parameter
     * - V2: Complex options struct with Create/Update/Release
     * - GENERIC: String-based key-value options
     */
    public function getApiType(): ProviderApiType
    {
        return match ($this) {
            self::CPU => ProviderApiType::LEGACY,
            self::CUDA => ProviderApiType::V2,
            self::ROCM => ProviderApiType::LEGACY,
            self::TENSORRT => ProviderApiType::V2,
            self::COREML => ProviderApiType::LEGACY,
            self::DIRECTML => ProviderApiType::LEGACY,
            self::OPENVINO => ProviderApiType::LEGACY,
            self::QNN => ProviderApiType::GENERIC,
            self::XNNPACK => ProviderApiType::GENERIC,
            self::DNNL => ProviderApiType::LEGACY,
            self::MIGRAPHX => ProviderApiType::LEGACY,
            self::WEBNN => ProviderApiType::GENERIC,
            self::WEBGPU => ProviderApiType::GENERIC,
            self::AZURE => ProviderApiType::GENERIC,
            self::VITIS_AI => ProviderApiType::GENERIC,
            self::JS => ProviderApiType::GENERIC,
        };
    }

    /**
     * Check if this provider is platform-specific.
     */
    public function isPlatformSpecific(): bool
    {
        return match ($this) {
            self::COREML => true,
            self::DIRECTML => true,
            self::QNN => true,
            default => false,
        };
    }

    /**
     * Get the supported platforms for this provider.
     *
     * @return array<string>
     */
    public function getSupportedPlatforms(): array
    {
        return match ($this) {
            self::CPU => ['linux', 'darwin', 'windows'],
            self::CUDA => ['linux', 'windows'],
            self::ROCM => ['linux'],
            self::TENSORRT => ['linux', 'windows'],
            self::COREML => ['darwin'],
            self::DIRECTML => ['windows'],
            self::OPENVINO => ['linux', 'darwin', 'windows'],
            self::QNN => ['linux', 'windows'],
            self::XNNPACK => ['linux', 'darwin', 'windows'],
            self::DNNL => ['linux', 'darwin', 'windows'],
            self::MIGRAPHX => ['linux'],
            self::WEBNN => [],
            self::WEBGPU => [],
            self::AZURE => ['linux', 'darwin', 'windows'],
            self::VITIS_AI => ['linux'],
            self::JS => [],
        };
    }

    /**
     * Get available providers that are actually compiled into the library.
     *
     * @return array<string>
     */
    public static function getAvailable(): array
    {
        try {
            return Lib::api()->getAvailableProviders();
        } catch (\Throwable $e) {
            return [self::CPU->value];
        }
    }

    /**
     * Get all execution provider values as strings.
     *
     * @return array<string>
     */
    public static function getAllValues(): array
    {
        return array_map(
            static fn (self $provider) => $provider->value,
            self::cases()
        );
    }
}

<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Enums;

/**
 * Controls whether you want to execute operators in the graph sequentially or in parallel.
 *
 * Usually when the model has many branches, setting this option to ExecutionMode::PARALLEL
 * will give you better performance.
 *
 * @see https://github.com/microsoft/onnxruntime/blob/main/docs/ONNX_Runtime_Perf_Tuning.md
 */
enum ExecutionMode: int
{
    case SEQUENTIAL = 0;
    case PARALLEL = 1;
}

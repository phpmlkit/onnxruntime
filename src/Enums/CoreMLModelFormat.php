<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Enums;

/**
 * CoreML model format options.
 *
 * Specifies whether to create a NeuralNetwork model or MLProgram model.
 *
 * @see https://developer.apple.com/documentation/coreml/mlmodel
 */
enum CoreMLModelFormat: string
{
    /**
     * Neural Network model format.
     * Traditional format, widely supported.
     */
    case NEURAL_NETWORK = 'NeuralNetwork';

    /**
     * MLProgram model format.
     * Newer format with better performance, requires Core ML 5 or later.
     */
    case ML_PROGRAM = 'MLProgram';
}

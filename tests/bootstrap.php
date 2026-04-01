<?php

declare(strict_types=1);

$projectRoot = dirname(__DIR__);
$modelsDir = $projectRoot.'/tests/Fixtures/models';
$identityModel = $modelsDir.'/identity.onnx';

// Generate ONNX test models on demand so fixture-dependent tests can run.
if (!file_exists($identityModel)) {
    $script = $projectRoot.'/scripts/generate_test_models.py';

    if (file_exists($script)) {
        $pythonBinaries = ['python3', 'python'];

        foreach ($pythonBinaries as $python) {
            $descriptorSpec = [
                1 => ['pipe', 'w'],
                2 => ['pipe', 'w'],
            ];

            $process = @proc_open([$python, $script], $descriptorSpec, $pipes, $projectRoot);

            if (!is_resource($process)) {
                continue;
            }

            $stdout = stream_get_contents($pipes[1]);
            $stderr = stream_get_contents($pipes[2]);

            fclose($pipes[1]);
            fclose($pipes[2]);

            $exitCode = proc_close($process);

            if (0 === $exitCode) {
                if (defined('STDERR')) {
                    fwrite(\STDERR, "[bootstrap] Generated ONNX test models.\n");
                }

                break;
            }
        }
    }
}

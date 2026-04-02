<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Tests;

use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;

/**
 * @internal
 *
 * @coversNothing
 */
final class ComposerReinstallRuntimeSwitchTest extends TestCase
{
    #[Test]
    public function composerReinstallSwitchesRuntimeArtifact(): void
    {
        if (!class_exists(\ZipArchive::class)) {
            $this->markTestSkipped('ZipArchive extension is required for integration test.');
        }

        $repoRoot = realpath(__DIR__.'/..');
        self::assertNotFalse($repoRoot);

        $tmpRoot = sys_get_temp_dir().\DIRECTORY_SEPARATOR.'onnxruntime-composer-int-'.uniqid('', true);
        $pluginCopyPath = $tmpRoot.\DIRECTORY_SEPARATOR.'platform-package-installer';
        $distPath = $tmpRoot.\DIRECTORY_SEPARATOR.'dist';
        $appPath = $tmpRoot.\DIRECTORY_SEPARATOR.'app';

        mkdir($tmpRoot, 0777, true);
        mkdir($distPath, 0777, true);
        mkdir($appPath, 0777, true);

        $server = null;

        try {
            $pluginSource = $repoRoot.'/vendor/codewithkyrian/platform-package-installer';
            $this->copyDirectory($pluginSource, $pluginCopyPath, ['vendor', 'tests', '.git', '.github']);
            $this->relaxPluginDependenciesForIntegration($pluginCopyPath, '2.1.0');

            $this->createZipArtifact($distPath.'/onnxruntime-1.2.3-runtime-cpu.zip', 'cpu');
            $this->createZipArtifact($distPath.'/onnxruntime-1.2.3-runtime-cuda12.zip', 'cuda12');

            $port = $this->reserveFreePort();
            $server = $this->startPhpServer($distPath, $port);
            $this->waitForServerReady("http://127.0.0.1:{$port}/onnxruntime-1.2.3-runtime-cpu.zip");

            $template = "http://127.0.0.1:{$port}/onnxruntime-{version}-runtime-{runtime}.zip";
            $fallbackUrl = "http://127.0.0.1:{$port}/onnxruntime-1.2.3-runtime-cpu.zip";

            $composerJson = [
                'name' => 'integration/onnxruntime-app',
                'type' => 'project',
                'require' => [
                    'codewithkyrian/platform-package-installer' => '2.1.0',
                    'phpmlkit/onnxruntime' => '1.2.3',
                ],
                'repositories' => [
                    ['packagist.org' => false],
                    [
                        'type' => 'path',
                        'url' => $pluginCopyPath,
                        'options' => ['symlink' => false],
                    ],
                    [
                        'type' => 'package',
                        'package' => [
                            'name' => 'phpmlkit/onnxruntime',
                            'version' => '1.2.3',
                            'type' => 'platform-package',
                            'dist' => [
                                'url' => $fallbackUrl,
                                'type' => 'zip',
                            ],
                            'extra' => [
                                'artifacts' => [
                                    'urls' => [
                                        'all' => $template,
                                    ],
                                    'vars' => [
                                        'runtime' => 'cpu',
                                    ],
                                ],
                            ],
                        ],
                    ],
                ],
                'config' => [
                    'allow-plugins' => [
                        'codewithkyrian/platform-package-installer' => true,
                    ],
                    'secure-http' => false,
                ],
            ];

            file_put_contents(
                $appPath.'/composer.json',
                json_encode($composerJson, \JSON_PRETTY_PRINT | \JSON_UNESCAPED_SLASHES)
            );

            $composerBin = $this->findComposerBinary($repoRoot);
            $env = [
                'COMPOSER_CACHE_DIR' => $tmpRoot.'/cache',
                'COMPOSER_HOME' => $tmpRoot.'/home',
            ];

            $install = $this->runCommand(
                [\PHP_BINARY, $composerBin, 'install', '--no-interaction', '--no-progress'],
                $appPath,
                $env,
                120
            );
            self::assertSame(0, $install['exit_code'], "composer install failed\n{$install['stderr']}\n{$install['stdout']}");
            self::assertSame('cpu', $this->readRuntimeMarker($appPath));

            $updatedJson = $composerJson;
            $updatedJson['extra'] = [
                'platform-packages' => [
                    'phpmlkit/onnxruntime' => [
                        'runtime' => 'cuda12',
                    ],
                ],
            ];

            file_put_contents(
                $appPath.'/composer.json',
                json_encode($updatedJson, \JSON_PRETTY_PRINT | \JSON_UNESCAPED_SLASHES)
            );

            $reinstall = $this->runCommand(
                [\PHP_BINARY, $composerBin, 'reinstall', 'phpmlkit/onnxruntime', '--no-interaction', '--no-progress'],
                $appPath,
                $env,
                120
            );
            self::assertSame(0, $reinstall['exit_code'], "composer reinstall failed\n{$reinstall['stderr']}\n{$reinstall['stdout']}");
            self::assertSame('cuda12', $this->readRuntimeMarker($appPath));
        } finally {
            if (\is_array($server)) {
                $this->stopProcess($server);
            }
            if (is_dir($tmpRoot)) {
                $this->removeDirectoryRecursive($tmpRoot);
            }
        }
    }

    private function readRuntimeMarker(string $appPath): string
    {
        $marker = $appPath.'/vendor/phpmlkit/onnxruntime/runtime-marker.txt';
        self::assertFileExists($marker, 'runtime marker file not found');

        return trim((string) file_get_contents($marker));
    }

    private function createZipArtifact(string $zipPath, string $runtime): void
    {
        $zip = new \ZipArchive();
        $ok = $zip->open($zipPath, \ZipArchive::CREATE | \ZipArchive::OVERWRITE);
        if (true !== $ok) {
            throw new \RuntimeException("Unable to create archive: {$zipPath}");
        }

        $zip->addFromString('runtime-marker.txt', $runtime.\PHP_EOL);
        $zip->addFromString('README.txt', 'onnxruntime integration artifact');
        $zip->close();
    }

    private function reserveFreePort(): int
    {
        $socket = stream_socket_server('tcp://127.0.0.1:0', $errno, $error);
        if (false === $socket) {
            throw new \RuntimeException("Unable to reserve a free port: {$error}");
        }

        $name = stream_socket_get_name($socket, false);
        fclose($socket);

        if (!\is_string($name) || !str_contains($name, ':')) {
            throw new \RuntimeException('Unable to read reserved port.');
        }

        return (int) substr($name, strrpos($name, ':') + 1);
    }

    /**
     * @return array{process: resource, stdout: resource, stderr: resource}
     */
    private function startPhpServer(string $documentRoot, int $port): array
    {
        $cmd = \sprintf(
            '%s -S %s -t %s',
            escapeshellarg(\PHP_BINARY),
            escapeshellarg("127.0.0.1:{$port}"),
            escapeshellarg($documentRoot)
        );

        $descriptorSpec = [
            0 => ['pipe', 'r'],
            1 => ['pipe', 'w'],
            2 => ['pipe', 'w'],
        ];

        $process = proc_open($cmd, $descriptorSpec, $pipes);
        if (!\is_resource($process)) {
            throw new \RuntimeException('Unable to start local PHP server.');
        }

        fclose($pipes[0]);
        stream_set_blocking($pipes[1], false);
        stream_set_blocking($pipes[2], false);

        return [
            'process' => $process,
            'stdout' => $pipes[1],
            'stderr' => $pipes[2],
        ];
    }

    private function waitForServerReady(string $url): void
    {
        for ($i = 0; $i < 20; ++$i) {
            usleep(100_000);
            $headers = @get_headers($url);
            if (\is_array($headers) && isset($headers[0]) && str_contains($headers[0], '200')) {
                return;
            }
        }

        throw new \RuntimeException("Local artifact server did not become ready for URL: {$url}");
    }

    /**
     * @param array<int, string>    $command
     * @param array<string, string> $env
     *
     * @return array{exit_code: int, stdout: string, stderr: string}
     */
    private function runCommand(array $command, string $cwd, array $env = [], int $timeoutSeconds = 60): array
    {
        $cmd = implode(' ', array_map('escapeshellarg', $command));
        $descriptorSpec = [
            0 => ['pipe', 'r'],
            1 => ['pipe', 'w'],
            2 => ['pipe', 'w'],
        ];

        $process = proc_open($cmd, $descriptorSpec, $pipes, $cwd, array_merge($_ENV, $env));
        if (!\is_resource($process)) {
            throw new \RuntimeException("Unable to run command: {$cmd}");
        }

        fclose($pipes[0]);

        $start = time();
        $finalStatus = null;
        while (true) {
            $status = proc_get_status($process);
            if (false === $status['running']) {
                $finalStatus = $status;

                break;
            }

            if ((time() - $start) > $timeoutSeconds) {
                proc_terminate($process);

                throw new \RuntimeException("Command timed out after {$timeoutSeconds}s: {$cmd}");
            }

            usleep(100_000);
        }

        $stdout = (string) stream_get_contents($pipes[1]);
        $stderr = (string) stream_get_contents($pipes[2]);
        fclose($pipes[1]);
        fclose($pipes[2]);

        $exitCode = proc_close($process);
        if (-1 === $exitCode && \is_array($finalStatus) && isset($finalStatus['exitcode']) && \is_int($finalStatus['exitcode'])) {
            $exitCode = $finalStatus['exitcode'];
        }

        return [
            'exit_code' => $exitCode,
            'stdout' => $stdout,
            'stderr' => $stderr,
        ];
    }

    /**
     * @param array{process: resource, stdout: resource, stderr: resource} $processData
     */
    private function stopProcess(array $processData): void
    {
        if (\is_resource($processData['stdout'])) {
            fclose($processData['stdout']);
        }
        if (\is_resource($processData['stderr'])) {
            fclose($processData['stderr']);
        }
        if (\is_resource($processData['process'])) {
            proc_terminate($processData['process']);
            proc_close($processData['process']);
        }
    }

    private function findComposerBinary(string $repoRoot): string
    {
        $localComposer = $repoRoot.'/vendor/bin/composer';
        if (file_exists($localComposer)) {
            return $localComposer;
        }

        $whichComposer = trim((string) shell_exec('which composer'));
        if ('' !== $whichComposer && file_exists($whichComposer)) {
            return $whichComposer;
        }

        throw new \RuntimeException('Composer binary not found. Ensure vendor/bin/composer or global composer is available.');
    }

    /**
     * @param array<int, string> $skipTopLevel
     */
    private function copyDirectory(string $sourceRoot, string $targetRoot, array $skipTopLevel = []): void
    {
        mkdir($targetRoot, 0777, true);

        $iterator = new \RecursiveIteratorIterator(
            new \RecursiveDirectoryIterator($sourceRoot, \FilesystemIterator::SKIP_DOTS),
            \RecursiveIteratorIterator::SELF_FIRST
        );

        foreach ($iterator as $item) {
            $relative = substr($item->getPathname(), \strlen($sourceRoot) + 1);
            $firstPart = explode(\DIRECTORY_SEPARATOR, $relative)[0];

            if (\in_array($firstPart, $skipTopLevel, true)) {
                continue;
            }

            $dest = $targetRoot.\DIRECTORY_SEPARATOR.$relative;
            if ($item->isDir()) {
                if (!is_dir($dest)) {
                    mkdir($dest, 0777, true);
                }

                continue;
            }

            copy($item->getPathname(), $dest);
        }
    }

    private function relaxPluginDependenciesForIntegration(string $pluginRoot, string $version): void
    {
        $composerPath = $pluginRoot.'/composer.json';
        $composer = json_decode((string) file_get_contents($composerPath), true);

        $composer['version'] = $version;
        $composer['require'] = [
            'php' => '^8.1',
            'composer-plugin-api' => '^1.1 || ^2.0',
            'composer-runtime-api' => '*',
        ];
        unset($composer['require-dev']);

        file_put_contents($composerPath, json_encode($composer, \JSON_PRETTY_PRINT | \JSON_UNESCAPED_SLASHES));
    }

    private function removeDirectoryRecursive(string $path): void
    {
        $iterator = new \RecursiveIteratorIterator(
            new \RecursiveDirectoryIterator($path, \FilesystemIterator::SKIP_DOTS),
            \RecursiveIteratorIterator::CHILD_FIRST
        );

        foreach ($iterator as $item) {
            if ($item->isDir()) {
                rmdir($item->getPathname());

                continue;
            }
            unlink($item->getPathname());
        }

        rmdir($path);
    }
}

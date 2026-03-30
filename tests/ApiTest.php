<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Tests;

use PhpMlKit\ONNXRuntime\Exception;
use PhpMlKit\ONNXRuntime\FFI\Lib;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;

class ApiTest extends TestCase
{
    protected function setUp(): void
    {
        if (!\extension_loaded('ffi')) {
            $this->markTestSkipped('FFI extension not available');
        }

        try {
            Lib::get();
        } catch (Exception $e) {
            $this->markTestSkipped('ONNX Runtime library not available: '.$e->getMessage());
        }
    }

    #[Test]
    public function itCanGetAvailableProviders(): void
    {
        $api = Lib::api();
        $providers = $api->getAvailableProviders();

        $this->assertIsArray($providers);
        $this->assertGreaterThanOrEqual(1, count($providers));
        $this->assertContains('CPUExecutionProvider', $providers);
    }
}
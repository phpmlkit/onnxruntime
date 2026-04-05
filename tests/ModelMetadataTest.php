<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Tests;

use PhpMlKit\ONNXRuntime\FFI\Lib;
use PhpMlKit\ONNXRuntime\InferenceSession;
use PhpMlKit\ONNXRuntime\Metadata\ModelMetadata;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;

/**
 * @internal
 */
#[CoversClass(ModelMetadata::class)]
class ModelMetadataTest extends TestCase
{
    private static string $fixturesDir;

    public static function setUpBeforeClass(): void
    {
        self::$fixturesDir = __DIR__.'/Fixtures/models';
    }

    protected function setUp(): void
    {
        if (!\extension_loaded('ffi')) {
            $this->markTestSkipped('FFI extension not available');
        }

        try {
            Lib::get();
        } catch (\Exception $e) {
            $this->markTestSkipped('ONNX Runtime library not available: '.$e->getMessage());
        }
    }

    #[Test]
    public function itCanGetModelMetadata(): void
    {
        $modelPath = self::$fixturesDir.'/identity.onnx';
        if (!file_exists($modelPath)) {
            $this->markTestSkipped('Test model not available');
        }

        $session = InferenceSession::fromFile($modelPath);
        $metadata = $session->metadata();

        $this->assertInstanceOf(ModelMetadata::class, $metadata);
    }

    #[Test]
    public function itLazyLoadsMetadata(): void
    {
        $modelPath = self::$fixturesDir.'/identity.onnx';
        if (!file_exists($modelPath)) {
            $this->markTestSkipped('Test model not available');
        }

        $session = InferenceSession::fromFile($modelPath);

        // First call should create the metadata
        $metadata1 = $session->metadata();
        $metadata2 = $session->metadata();

        // Should be the same instance (lazy loaded and cached)
        $this->assertSame($metadata1, $metadata2);
    }

    #[Test]
    public function itExtractsProducerName(): void
    {
        $modelPath = self::$fixturesDir.'/identity.onnx';
        if (!file_exists($modelPath)) {
            $this->markTestSkipped('Test model not available');
        }

        $session = InferenceSession::fromFile($modelPath);
        $metadata = $session->metadata();

        $this->assertEquals('onnx-test-generator', $metadata->getProducerName());
    }

    #[Test]
    public function itExtractsGraphName(): void
    {
        $modelPath = self::$fixturesDir.'/identity.onnx';
        if (!file_exists($modelPath)) {
            $this->markTestSkipped('Test model not available');
        }

        $session = InferenceSession::fromFile($modelPath);
        $metadata = $session->metadata();

        $this->assertEquals('identity_graph', $metadata->getGraphName());
    }

    #[Test]
    public function itExtractsDomain(): void
    {
        $modelPath = self::$fixturesDir.'/identity.onnx';
        if (!file_exists($modelPath)) {
            $this->markTestSkipped('Test model not available');
        }

        $session = InferenceSession::fromFile($modelPath);
        $metadata = $session->metadata();

        $this->assertEquals('test.onnxruntime-php', $metadata->getDomain());
    }

    #[Test]
    public function itExtractsDescription(): void
    {
        $modelPath = self::$fixturesDir.'/identity.onnx';
        if (!file_exists($modelPath)) {
            $this->markTestSkipped('Test model not available');
        }

        $session = InferenceSession::fromFile($modelPath);
        $metadata = $session->metadata();

        $this->assertEquals('Identity model for testing', $metadata->getDescription());
    }

    #[Test]
    public function itExtractsVersion(): void
    {
        $modelPath = self::$fixturesDir.'/identity.onnx';
        if (!file_exists($modelPath)) {
            $this->markTestSkipped('Test model not available');
        }

        $session = InferenceSession::fromFile($modelPath);
        $metadata = $session->metadata();

        $this->assertEquals(1, $metadata->getVersion());
    }

    #[Test]
    public function itExtractsCustomMetadataMap(): void
    {
        $modelPath = self::$fixturesDir.'/identity.onnx';
        if (!file_exists($modelPath)) {
            $this->markTestSkipped('Test model not available');
        }

        $session = InferenceSession::fromFile($modelPath);
        $metadata = $session->metadata();

        $customMap = $metadata->getCustomMetadataMap();

        $this->assertIsArray($customMap);
        $this->assertArrayHasKey('author', $customMap);
        $this->assertArrayHasKey('description', $customMap);
        $this->assertArrayHasKey('framework', $customMap);
        $this->assertArrayHasKey('test_category', $customMap);

        $this->assertEquals('PhpMlKit\ONNXRuntime Test Suite', $customMap['author']);
        $this->assertEquals('Simple identity operation test model', $customMap['description']);
        $this->assertEquals('ONNX', $customMap['framework']);
        $this->assertEquals('basic', $customMap['test_category']);
    }

    #[Test]
    public function itReturnsEmptyStringsForModelsWithoutMetadata(): void
    {
        $modelPath = self::$fixturesDir.'/add.onnx';
        if (!file_exists($modelPath)) {
            $this->markTestSkipped('Test model not available');
        }

        $session = InferenceSession::fromFile($modelPath);
        $metadata = $session->metadata();

        // Models without explicit metadata should return empty strings
        $this->assertIsString($metadata->getProducerName());
        $this->assertIsString($metadata->getGraphName());
        $this->assertIsString($metadata->getDomain());
        $this->assertIsString($metadata->getDescription());
        $this->assertIsString($metadata->getGraphDescription());
        $this->assertIsInt($metadata->getVersion());
        $this->assertIsArray($metadata->getCustomMetadataMap());
    }
}

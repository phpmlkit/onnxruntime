<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Metadata;

use FFI\CData;
use PhpMlKit\ONNXRuntime\FFI\Api;
use PhpMlKit\ONNXRuntime\FFI\Lib;
use PhpMlKit\ONNXRuntime\InferenceSession;

/**
 * Model metadata containing information about the ONNX model.
 *
 * This class provides access to model metadata such as producer name,
 * graph name, domain, description, version, and custom metadata.
 * The metadata is extracted from the model during construction.
 */
final class ModelMetadata
{
    /** @var string Producer name */
    private readonly string $producerName;

    /** @var string Graph name */
    private readonly string $graphName;

    /** @var string Domain */
    private readonly string $domain;

    /** @var string Description */
    private readonly string $description;

    /** @var string Graph description */
    private readonly string $graphDescription;

    /** @var int Version */
    private readonly int $version;

    /** @var array<string, string> Custom metadata key-value pairs */
    private readonly array $customMetadataMap;

    /**
     * Extract model metadata from an InferenceSession.
     *
     * @param InferenceSession $session The inference session to extract metadata from
     */
    public function __construct(InferenceSession $session)
    {
        $api = Lib::api();

        $modelMetadata = $api->sessionGetModelMetadata($session->getHandle());
        $allocator = $session->getAllocator();

        try {
            $producerNamePtr = $api->modelMetadataGetProducerName($modelMetadata, $allocator);
            $this->producerName = $this->stringFromNativeUtf8($producerNamePtr);
            $api->allocatorFree($allocator, $producerNamePtr);

            $graphNamePtr = $api->modelMetadataGetGraphName($modelMetadata, $allocator);
            $this->graphName = $this->stringFromNativeUtf8($graphNamePtr);
            $api->allocatorFree($allocator, $graphNamePtr);

            $domainPtr = $api->modelMetadataGetDomain($modelMetadata, $allocator);
            $this->domain = $this->stringFromNativeUtf8($domainPtr);
            $api->allocatorFree($allocator, $domainPtr);

            $descriptionPtr = $api->modelMetadataGetDescription($modelMetadata, $allocator);
            $this->description = $this->stringFromNativeUtf8($descriptionPtr);
            $api->allocatorFree($allocator, $descriptionPtr);

            $graphDescriptionPtr = $api->modelMetadataGetGraphDescription($modelMetadata, $allocator);
            $this->graphDescription = $this->stringFromNativeUtf8($graphDescriptionPtr);
            $api->allocatorFree($allocator, $graphDescriptionPtr);

            $this->version = $api->modelMetadataGetVersion($modelMetadata);

            $this->customMetadataMap = $this->extractCustomMetadataMap($api, $modelMetadata, $allocator);
        } finally {
            $api->releaseModelMetadata($modelMetadata);
        }
    }

    /**
     * Get the producer name of the model.
     */
    public function getProducerName(): string
    {
        return $this->producerName;
    }

    /**
     * Get the graph name of the model.
     */
    public function getGraphName(): string
    {
        return $this->graphName;
    }

    /**
     * Get the domain of the model.
     */
    public function getDomain(): string
    {
        return $this->domain;
    }

    /**
     * Get the description of the model.
     */
    public function getDescription(): string
    {
        return $this->description;
    }

    /**
     * Get the graph description of the model.
     */
    public function getGraphDescription(): string
    {
        return $this->graphDescription;
    }

    /**
     * Get the version of the model.
     */
    public function getVersion(): int
    {
        return $this->version;
    }

    /**
     * Get the custom metadata map.
     *
     * @return array<string, string> Key-value pairs of custom metadata
     */
    public function getCustomMetadataMap(): array
    {
        return $this->customMetadataMap;
    }

    /**
     * Convert native UTF-8 string to PHP string.
     */
    private function stringFromNativeUtf8(CData $ptr): string
    {
        if (null === $ptr) {
            return '';
        }

        return \FFI::string($ptr);
    }

    /**
     * Extract custom metadata map from model metadata.
     *
     * @return array<string, string>
     */
    private function extractCustomMetadataMap(Api $api, CData $modelMetadata, CData $allocator): array
    {
        $customMetadata = [];

        [$keysPtr, $numKeys] = $api->modelMetadataGetCustomMetadataMapKeys($modelMetadata, $allocator);

        if (0 === $numKeys) {
            $api->allocatorFree($allocator, $keysPtr);

            return $customMetadata;
        }

        try {
            for ($i = 0; $i < $numKeys; ++$i) {
                $keyPtr = $keysPtr[$i];
                $key = $this->stringFromNativeUtf8($keyPtr);

                $valuePtr = $api->modelMetadataLookupCustomMetadataMap($modelMetadata, $allocator, $key);
                $value = $this->stringFromNativeUtf8($valuePtr);
                $api->allocatorFree($allocator, $valuePtr);

                $customMetadata[$key] = $value;
            }
        } finally {
            $api->allocatorFree($allocator, $keysPtr);
        }

        return $customMetadata;
    }
}

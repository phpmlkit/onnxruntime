<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\FFI;

use Codewithkyrian\PlatformPackageInstaller\Platform;
use FFI;
use FFI\CData;
use PhpMlKit\ONNXRuntime\Enums\ErrorCode;

/**
 * FFI Singleton for ONNX Runtime library using the proper C API approach.
 *
 * ONNX Runtime C API works by:
 * 1. Calling OrtGetApiBase() to get an OrtApiBase*
 * 2. Calling base->GetApi(version) to get an OrtApi* struct
 * 3. Accessing all functions via function pointers in the OrtApi struct
 */
final class Lib
{
    private const PLATFORM_CONFIGS = [
        'linux-x86_64' => [
            'directory' => 'linux-x86_64',
            'libraryTemplate' => 'libonnxruntime.so.{version}',
        ],
        'linux-arm64' => [
            'directory' => 'linux-arm64',
            'libraryTemplate' => 'libonnxruntime.so.{version}',
        ],
        'darwin-arm64' => [
            'directory' => 'darwin-arm64',
            'libraryTemplate' => 'libonnxruntime.{version}.dylib',
        ],
        'windows-x64' => [
            'directory' => 'windows-x64',
            'libraryTemplate' => 'onnxruntime.dll',
        ],
    ];

    private const VERSION = '1.24.3';
    private const API_VERSION = 24;

    private static ?\FFI $ffi = null;
    private static ?Api $api = null;
    private static ?string $libraryPath = null;
    private static $libc;

    /**
     * Get FFI instance (singleton).
     */
    public static function get(): \FFI
    {
        if (null === self::$ffi) {
            self::$ffi = self::create();
        }

        return self::$ffi;
    }

    /**
     * Get the typed API wrapper.
     */
    public static function api(): Api
    {
        if (null === self::$api) {
            self::get();
            $base = self::get()->OrtGetApiBase()[0];
            $apiPtr = ($base->GetApi)(self::API_VERSION)[0];
            self::$api = new Api(self::$ffi, $apiPtr);
        }

        return self::$api;
    }

    /**
     * Get the path to the loaded library.
     */
    public static function getLibraryPath(): string
    {
        if (null === self::$libraryPath) {
            self::get();
        }

        return self::$libraryPath;
    }

    /**
     * Check FFI status and throw specific exception on error.
     *
     * Maps ONNX Runtime error codes to specific exception classes for better
     * error handling and debugging.
     */
    public static function checkStatus(?CData $status): void
    {
        if (null === $status) {
            return;
        }

        $api = self::api();
        $errorCode = $api->getErrorCode($status);
        $message = $api->getErrorMessage($status);
        $api->releaseStatus($status);

        $errorCodeEnum = ErrorCode::from($errorCode);
        $exceptionClass = $errorCodeEnum->getExceptionClass();

        throw new $exceptionClass($message);
    }

    // for Windows
    public static function libc()
    {
        if (!isset(self::$libc)) {
            self::$libc = \FFI::cdef(
                'size_t mbstowcs(void *wcstr, const char *mbstr, size_t count);',
                'msvcrt.dll'
            );
        }

        return self::$libc;
    }

    /**
     * Convert a PHP string into the proper ONNX Runtime path string type.
     *
     * ONNX Runtime uses ORTCHAR_T for paths:
     * - Windows: UTF-16 (wide)
     * - Others:  UTF-8 (char*)
     *
     * @return string|CData
     */
    public static function ortString(string $value): string|CData
    {
        if (\PHP_OS_FAMILY !== 'Windows') {
            return $value;
        }

        $libc = self::libc();
        $strlen = \strlen($value);
        $maxChars = $strlen + 1;

        $dest = $libc->new('char[' . ($maxChars * 2) . ']');
        $ret = (int) $libc->mbstowcs($dest, $value, $maxChars);

        if ($ret != $strlen) {
            throw new \RuntimeException('Expected mbstowcs to return ' . $strlen . ", got $ret");
        }

        return $dest;
    }

    private static function create(): \FFI
    {
        $platform = self::detectPlatform();
        $config = self::PLATFORM_CONFIGS[$platform] ?? null;

        if (null === $config) {
            throw new \RuntimeException("Unsupported platform: {$platform}");
        }

        $libraryPath = self::findLibrary($config);

        if (null === $libraryPath) {
            throw new \RuntimeException(
                "ONNX Runtime library not found for platform {$platform}. "
                    .'Please run: composer run post-install-cmd'
            );
        }

        self::$libraryPath = $libraryPath;

        $header = self::getCHeader();

        return \FFI::cdef($header, $libraryPath);
    }

    private static function detectPlatform(): string
    {
        $platform = Platform::findBestMatch(self::PLATFORM_CONFIGS);
        if (false === $platform) {
            throw new \RuntimeException('Could not detect platform');
        }

        return $platform['directory'];
    }

    private static function findLibrary(array $config): ?string
    {
        $libDir = __DIR__.'/../../lib/'.$config['directory'].'/lib';

        // Replace {version} in template
        $libraryName = str_replace('{version}', self::VERSION, $config['libraryTemplate']);
        $libraryPath = $libDir.'/'.$libraryName;

        if (file_exists($libraryPath)) {
            return $libraryPath;
        }

        // Fallback: search for any matching library
        $pattern = str_replace('{version}', '*', $config['libraryTemplate']);
        $matches = glob($libDir.'/'.$pattern);

        return $matches[0] ?? null;
    }

    /**
     * Get C header for FFI from include/onnxruntime.h.
     */
    private static function getCHeader(): string
    {
        $headerPath = __DIR__.'/../../include/onnxruntime.h';
        $header = file_get_contents($headerPath);

        if (false === $header) {
            throw new \RuntimeException("Could not read header: {$headerPath}");
        }

        if (\PHP_OS_FAMILY == 'Darwin') {
            $coremlHeaderPath = __DIR__.'/../../include/onnxruntime_coreml_provider.h';
            $header .= "\n" . file_get_contents($coremlHeaderPath);
        }

        if (\PHP_OS_FAMILY == 'Windows') {
            $dmlHeaderPath = __DIR__.'/../../include/onnxruntime_dml_provider.h';
            $header .= "\n" . file_get_contents($dmlHeaderPath);
        }

        return $header;
    }
}

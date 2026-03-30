# ONNX Runtime for PHP (CPU)

High-performance ONNX Runtime bindings for PHP using CPU execution.

> **⚠️ This is a read-only split repository.**
> 
> For full documentation, contributing guidelines, bug reports, and feature requests, please visit the main repository:
> **https://github.com/phpmlkit/onnxruntime**

## Installation

```bash
composer require phpmlkit/onnxruntime
```

## Quick Start

```php
use PhpMlKit\ONNXRuntime\InferenceSession;
use PhpMlKit\ONNXRuntime\OrtValue;
use PhpMlKit\ONNXRuntime\Enums\DataType;

$session = InferenceSession::fromFile('model.onnx');
$input = OrtValue::fromArray([1.0, 2.0, 3.0], DataType::FLOAT);
$outputs = $session->run(['input' => $input]);
$result = $outputs['output']->toArray();
```

## Documentation

- **Full Documentation**: https://github.com/phpmlkit/onnxruntime/blob/main/README.md
- **API Reference**: See inline PHPDoc in the source code
- **Examples**: https://github.com/phpmlkit/onnxruntime/tree/main/examples

## GPU Support

Need GPU acceleration? Use the GPU package instead:

```bash
composer require phpmlkit/onnxruntime-gpu
```

## Contributing

Contributions are welcome! Please submit pull requests to the main repository:
https://github.com/phpmlkit/onnxruntime

## Support

- **Issues**: https://github.com/phpmlkit/onnxruntime/issues
- **Discussions**: https://github.com/phpmlkit/onnxruntime/discussions

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This package is automatically generated from the monorepo. Do not submit pull requests to this repository.
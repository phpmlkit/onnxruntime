# Monorepo Structure

This repository is a monorepo containing multiple ONNX Runtime PHP packages.

## Packages

| Package | Repository | Description | Install |
|---------|-----------|-------------|---------|
| `phpmlkit/onnxruntime` | [phpmlkit/onnxruntime-cpu](https://github.com/phpmlkit/onnxruntime-cpu) | CPU-only version (default) | `composer require phpmlkit/onnxruntime` |
| `phpmlkit/onnxruntime-gpu` | [phpmlkit/onnxruntime-gpu](https://github.com/phpmlkit/onnxruntime-gpu) | GPU/CUDA support | `composer require phpmlkit/onnxruntime-gpu` |

## Development Workflow

### Making Changes

1. Work in the root directory on the `main` branch
2. All source code is in `src/` (shared across all packages)
3. Package-specific configuration is in `packages/{package-name}/`

### Local Testing

```bash
# Prepare packages (copies shared source to package directories)
composer monorepo:prepare

# Or run directly
bash scripts/prepare-packages.sh
```

**Note:** The `packages/` directory contents are gitignored (except `composer.json`). Don't commit the prepared packages locally - they're generated during the CI split workflow.

### Automated Splitting

When you push to `main` or create a tag:

1. GitHub Action runs `prepare-packages.sh`
2. Copies shared source (`src/`, `scripts/`, etc.) to each package
3. Splits each package directory to its respective repository using splitsh-lite
4. Tags are automatically synchronized

## Repository Naming

- **Main repo** (this one): `phpmlkit/onnxruntime` - Development monorepo (not installable)
- **CPU repo**: `phpmlkit/onnxruntime-cpu` - Contains CPU-only package
- **GPU repo**: `phpmlkit/onnxruntime-gpu` - Contains GPU/CUDA package

The Composer package names are clean:
- `phpmlkit/onnxruntime` (CPU/default)
- `phpmlkit/onnxruntime-gpu` (GPU)

## Adding a New Package

1. Create directory: `packages/onnxruntime-{name}/`
2. Add `composer.json` with appropriate `conflict` rules
3. Update `.github/workflows/split.yml`
4. Update `scripts/prepare-packages.sh`
5. Update `.gitignore`
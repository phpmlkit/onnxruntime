#!/bin/bash
# Prepare packages for split by copying shared source code

set -e

echo "Preparing packages for split..."

SHARED_ITEMS=(
    "src"
    "scripts"
    "include"
    "LICENSE"
    ".php-cs-fixer.php"
    "phpunit.xml"
)

copy_to_package() {
    local package_dir=$1
    
    echo "Copying shared files to ${package_dir}..."
    
    for item in "${SHARED_ITEMS[@]}"; do
        if [ -e "$item" ]; then
            if [ -e "${package_dir}/${item}" ]; then
                rm -rf "${package_dir}/${item}"
            fi
            
            cp -r "$item" "${package_dir}/"
        fi
    done
}

copy_to_package "packages/onnxruntime-cpu"
copy_to_package "packages/onnxruntime-gpu"
copy_to_package "packages/onnxruntime-directml"

echo "Packages prepared for split."
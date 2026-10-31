#!/bin/sh
# Join split zip parts and unzip the abliterated checkpoint.
set -e
cat Qwen2.5-0.5B-Abliterated.zip.part-* > Qwen2.5-0.5B-Abliterated.zip
sha256sum Qwen2.5-0.5B-Abliterated.zip
echo "expected: 82ca983efee4c3be631ab40a4566dd2b4659b742813d7bfe1481749cf7d017c1"
unzip Qwen2.5-0.5B-Abliterated.zip
echo "Checkpoint in qwen-abliterated-bf16/"

#!/usr/bin/env bash
# Download the three benchmarks into data/ (they are not redistributed here; see their licenses).
set -euo pipefail
mkdir -p data/socialmembench
# LoCoMo (CC BY-NC 4.0)
curl -L -o data/locomo10.json https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
# LongMemEval-S, cleaned release (MIT), about 277 MB
curl -L -o data/longmemeval_s_cleaned.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
# SocialMemBench (CC BY 4.0), the revision used in the paper
REV=97d38bc57349b3aed1236904968af1df89539c48
for f in conversations.parquet qa.parquet; do
  curl -L -o "data/socialmembench/$f" "https://huggingface.co/datasets/anon4data/socialmembench/resolve/$REV/$f"
done

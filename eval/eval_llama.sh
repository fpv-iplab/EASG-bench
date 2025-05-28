#!/bin/bash

context="clip_narrations.json"
rule="rule.txt"
result="results/Qwen2.5-VL-7B-Instruct"
scale="scale.json"

python eval_llama_review.py \
	--context "${context}" \
	--rule "${rule}" \
	--answer "${result}/answers.json" \
	--output "${result}/scores.json" \
	--scale "${scale}"


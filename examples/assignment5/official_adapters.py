"""Adapters that connect the example implementation to the official tests."""

from cs336_alignment.data import PackedSFTDataset, iterate_batches
from cs336_alignment.dpo import compute_per_instance_dpo_loss
from cs336_alignment.grpo import (
    aggregate_loss_across_microbatch,
    compute_group_normalized_rewards,
    compute_policy_gradient_loss,
    compute_rollout_rewards,
    get_response_log_probs,
    grpo_train_step,
    tokenize_prompt_and_output,
)
from cs336_alignment.metrics import parse_gsm8k_response, parse_mmlu_response


run_tokenize_prompt_and_output = tokenize_prompt_and_output
run_get_response_log_probs = get_response_log_probs
run_compute_rollout_rewards = compute_rollout_rewards
run_compute_group_normalized_rewards = compute_group_normalized_rewards
run_compute_policy_gradient_loss = compute_policy_gradient_loss
run_aggregate_loss_across_microbatch = aggregate_loss_across_microbatch
run_grpo_train_step = grpo_train_step
get_packed_sft_dataset = PackedSFTDataset
run_iterate_batches = iterate_batches
run_parse_mmlu_response = parse_mmlu_response
run_parse_gsm8k_response = parse_gsm8k_response
run_compute_per_instance_dpo_loss = compute_per_instance_dpo_loss

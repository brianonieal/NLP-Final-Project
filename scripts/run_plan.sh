#!/usr/bin/env bash
# GenAI disclosure: Claude (Anthropic) generated this file. Brian Onieal reviewed it.
#
# Runs one piece of the experiment plan on a single-GPU Linux machine.
# Every piece can be rerun: training resumes from its last checkpoint and
# evaluation skips checkpoints that already have results.
#
#   bash scripts/run_plan.sh timing          one step with thinking on, one with it off
#   bash scripts/run_plan.sh sdpo            Experiment 1 and the SDPO row of Experiment 2
#   bash scripts/run_plan.sh sdpo_kl         Experiment 2: KL penalty baseline
#   bash scripts/run_plan.sh sdpo_gram 1.0   Experiment 2: Gram anchoring at weight 1.0
#   bash scripts/run_plan.sh report          one table for every run
#
# Settings come from environment variables:
#   RUNS=runs            where runs are written
#   THINKING=false       Qwen3 thinking mode for the real runs
#   ROLLOUT_BATCH=8      prompts per sampling call during training
#   EVAL_BATCH=2         prompts per sampling call during evaluation
#   SDPO_CHECKPOINTS     which checkpoints the sdpo piece evaluates
set -euo pipefail

RUNS="${RUNS:-runs}"
THINKING="${THINKING:-false}"
SHARED="rollout.batch_size=${ROLLOUT_BATCH:-8} eval.batch_size=${EVAL_BATCH:-2} \
train.save_every=5 eval.probe_file=${RUNS}/probes.json model.enable_thinking=${THINKING}"

train_and_eval() {
    # $1 config, $2 run name, $3 checkpoints to evaluate, $4 extra overrides
    local settings="--override ${SHARED} train.output_dir=${RUNS}/$2 ${4:-}"
    python -m gram_sdpo.train --config "$1" --resume ${settings}
    python -m gram_sdpo.evaluate --config "$1" --checkpoints $3 ${settings}
}

case "${1:-}" in
    timing)
        for mode in true false; do
            name="timing_thinking_$([ "${mode}" = true ] && echo on || echo off)"
            python -m gram_sdpo.train --config configs/colab_timing.yaml --override \
                "rollout.batch_size=${ROLLOUT_BATCH:-8}" "model.enable_thinking=${mode}" \
                "train.output_dir=${RUNS}/${name}"
        done
        python -m gram_sdpo.timing --runs "${RUNS}" --plan configs/sdpo.yaml
        ;;
    sdpo)
        # Checkpoints are saved every 5 steps so a dropped run loses little.
        # Only these are evaluated, because each evaluation takes hours.
        train_and_eval configs/sdpo.yaml sdpo \
            "${SDPO_CHECKPOINTS:-base step-00020 step-00040 step-00060 step-00080 step-00100}"
        ;;
    sdpo_kl)
        train_and_eval configs/sdpo_kl.yaml sdpo_kl final
        ;;
    sdpo_gram)
        weight="${2:?give the Gram weight, for example: sdpo_gram 1.0}"
        train_and_eval configs/sdpo_gram.yaml "sdpo_gram_w${weight}" final "loss.gram_weight=${weight}"
        ;;
    report)
        python -m gram_sdpo.report --runs "${RUNS}"
        cat "${RUNS}/results.md"
        ;;
    *)
        sed -n '4,20p' "$0"
        exit 1
        ;;
esac

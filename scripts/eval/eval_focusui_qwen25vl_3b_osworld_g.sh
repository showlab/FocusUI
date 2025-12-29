
UI_GROUNDING_BENCH_BASE_DIR="datasets/ui_benchmarks"

MODEL_TYPE="focusui_guiactor_3b_qwen25vl"
MODEL_PATH="checkpoints/focusui_guiactor_3b_qwen25vl"
SAVE_PATH="eval_results/focusui_guiactor_3b_qwen25vl/osworld_g"

EVAL_VISUAL_REDUCT_RATIOS="0.00 0.20 0.40 0.50 0.60 0.70 0.80 0.90"

for drop in $EVAL_VISUAL_REDUCT_RATIOS; do
    echo "OSWorld-G | drop=${drop} | device=cuda:0"
    python -m evaluation.os_world_g_eval \
        --model_type "${MODEL_TYPE}" \
        --model_name_or_path "${MODEL_PATH}" \
        --save_path "${SAVE_PATH}/drop_${drop}" \
        --data_path "${UI_GROUNDING_BENCH_BASE_DIR}/osworld_g" \
        --topk 3 \
        --device "cuda:0" \
        --visual_reduct_ratio "${drop}" \
        --num_overlay_samples 0
done



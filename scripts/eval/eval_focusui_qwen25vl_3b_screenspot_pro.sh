
UI_GROUNDING_BENCH_BASE_DIR="datasets/ui_benchmarks"

MODEL_TYPE="focusui_guiactor_3b_qwen25vl"
MODEL_PATH="checkpoints/focusui_guiactor_3b_qwen25vl"
SAVE_PATH="eval_results/focusui_guiactor_3b_qwen25vl/screenspot_pro"

EVAL_VISUAL_REDUCT_RATIOS="0.00 0.20 0.40 0.50 0.60 0.70 0.80 0.90"

for drop in $EVAL_VISUAL_REDUCT_RATIOS; do
    echo "ScreenSpot-Pro | drop=${drop} | device=cuda:0"
    python -m evaluation.ss_pro_eval \
        --model_type "${MODEL_TYPE}" \
        --model_name_or_path "${MODEL_PATH}" \
        --save_path "${SAVE_PATH}/drop_${drop}" \
        --data_path "${UI_GROUNDING_BENCH_BASE_DIR}/ScreenSpot-Pro_HF" \
        --topk 3 \
        --device "cuda:0" \
        --visual_reduct_ratio "${drop}" \
        --num_overlay_samples 0
done



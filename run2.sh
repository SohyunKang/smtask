#!/bin/bash

FOLDS=5
START_SEED=41
END_SEED=49
GPU_ID=0

# ✅ 모델과 데이터 이름 변수로 지정
MODEL_NAME="attention_unet"
DATA_NAME="orig"

for ((j=START_SEED; j<=END_SEED; j++)); do
    for ((i=0; i<$FOLDS; i++)); do
        echo "🚀 [Seed $j | Fold $i | Model $MODEL_NAME | Data $DATA_NAME] 시작합니다..."

        CUDA_VISIBLE_DEVICES=$GPU_ID python main_orig_attunet_pt.py \
            --fold $i \
            --data $DATA_NAME \
            --model $MODEL_NAME \
            --seed $j

        EXIT_CODE=$?
        if [ $EXIT_CODE -ne 0 ]; then
            echo "❌ 오류 발생 (Seed $j | Fold $i | Model $MODEL_NAME | Data $DATA_NAME | Exit Code $EXIT_CODE)" >> failed_log.txt
        else
            echo "✅ [Seed $j | Fold $i] 완료!"
        fi
        echo "-----------------------------"
    done
done

echo "🎉 모든 실험 완료: $START_SEED ~ $END_SEED, 모델=$MODEL_NAME, 데이터=$DATA_NAME"

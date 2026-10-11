#!/bin/bash
# ============================================================================
# FinEduBench-Luso: Pipeline completo de experimentos
#
# Executa:
#   1. Benchmark multi-modelo (baseline, sem treinamento)
#   2. LoRA fine-tuning (sem cross-lingual)
#   3. LoRA fine-tuning (com cross-lingual)
#   4. Analise comparativa final
#
# Uso:
#   chmod +x run_experiments.sh
#   ./run_experiments.sh          # pipeline completo
#   ./run_experiments.sh benchmark # so benchmark
#   ./run_experiments.sh train     # so treinamentos
#   ./run_experiments.sh compare   # so comparacao
# ============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "============================================"
echo " FinEduBench-Luso: Pipeline de Experimentos"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo ""

STEP="${1:-all}"

# -----------------------------------------------------------
# 1. Benchmark multi-modelo (modelos pre-treinados)
# -----------------------------------------------------------
if [[ "$STEP" == "all" || "$STEP" == "benchmark" ]]; then
    echo ">>> [1/4] Benchmark multi-modelo..."
    echo "    Avalia: OpenCLIP, XLM-R CLIP, NLLB-CLIP, e5-multilingual"
    echo ""
    python3 -m embeddings.model_benchmark
    echo ""
    echo ">>> Benchmark concluido."
    echo ""
fi

# -----------------------------------------------------------
# 2. LoRA fine-tuning (sem cross-lingual)
# -----------------------------------------------------------
if [[ "$STEP" == "all" || "$STEP" == "train" ]]; then
    echo ">>> [2/4] LoRA fine-tuning (sem cross-lingual)..."
    echo ""
    python3 -m embeddings.clip_lora_finetune \
        --epochs 20 \
        --rank 4 \
        --alpha 8.0 \
        --batch-size 32 \
        --lr 1e-4 \
        --output-dir embeddings/lora_output
    echo ""
    echo ">>> LoRA fine-tuning concluido."
    echo ""

    # -----------------------------------------------------------
    # 3. LoRA fine-tuning (com cross-lingual)
    # -----------------------------------------------------------
    echo ">>> [3/4] LoRA fine-tuning + Cross-Lingual..."
    echo ""
    python3 -m embeddings.clip_lora_finetune \
        --epochs 20 \
        --rank 4 \
        --alpha 8.0 \
        --batch-size 32 \
        --lr 1e-4 \
        --cross-lingual \
        --cl-weight 0.3 \
        --output-dir embeddings/lora_output_cl
    echo ""
    echo ">>> LoRA+CL fine-tuning concluido."
    echo ""
fi

# -----------------------------------------------------------
# 4. Analise comparativa (Baseline vs LoRA vs LoRA+CL)
# -----------------------------------------------------------
if [[ "$STEP" == "all" || "$STEP" == "compare" ]]; then
    echo ">>> [4/4] Analise comparativa..."
    echo ""
    python3 -m embeddings.clip_compare
    echo ""
    echo ">>> Analise comparativa concluida."
    echo ""
fi

echo "============================================"
echo " Pipeline concluido!"
echo " Resultados em: embeddings/results/"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

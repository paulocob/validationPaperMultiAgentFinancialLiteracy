#!/usr/bin/env bash
# ============================================================================
# FinEduBench-Luso: Pipeline completo de geracao, analise e avaliacao
#
# Benchmark multimodal para educacao financeira lusofona (pt-BR, pt-PT, pt-AO)
# Pesquisa de doutorado - UFCG
#
# Uso: bash generate_and_evaluate.sh
# Requisitos: Python 3.12+, dependencias em requirements.txt
# ============================================================================

set -euo pipefail

# Diretorio raiz do projeto
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo " FinEduBench-Luso - Pipeline Completo"
echo "============================================"
echo ""

# Verifica se dependencias estao instaladas
echo "[0/4] Verificando dependencias..."
python3 -c "import rich; import pandas; import numpy" 2>/dev/null || {
    echo "ERRO: Dependencias nao instaladas. Execute:"
    echo "  pip install -r requirements.txt"
    exit 1
}
echo "  Dependencias OK."
echo ""

# 1. Gerar dataset
echo "============================================"
echo "[1/4] Gerando dataset semente..."
echo "============================================"
python3 -m dataset.generator
echo ""

# 2. Executar analise CLIP
echo "============================================"
echo "[2/4] Executando analise CLIP (Lusofono-CLIP-Bench)..."
echo "============================================"
python3 -m embeddings.clip_analysis
echo ""

# 3. Executar baselines RAG
echo "============================================"
echo "[3/4] Executando baselines RAG..."
echo "============================================"
python3 -m baselines.run_baselines
echo ""

# 4. Avaliar baselines
echo "============================================"
echo "[4/4] Avaliando baselines..."
echo "============================================"
python3 -m evaluation.run_evaluation --predictions baselines/results/
echo ""

echo "============================================"
echo " Pipeline concluido!"
echo ""
echo " Resultados disponiveis em:"
echo "   - dataset/BR/, dataset/PT/, dataset/AO/  (dados gerados)"
echo "   - embeddings/results/                     (analise CLIP)"
echo "   - baselines/results/                      (predicoes dos baselines)"
echo "   - evaluation/results/                     (metricas e tabelas LaTeX)"
echo "============================================"

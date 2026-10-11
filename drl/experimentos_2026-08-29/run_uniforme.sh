#!/bin/bash
# 4 algoritmos x 5 sementes, recompensa UNIFORME (m6=1.0 onde a flag existe)
for s in 42 43 44 45 46; do
  for alg in dqn ppo cql bc; do
    EXTRA=""
    if [ "$alg" = "dqn" ] || [ "$alg" = "ppo" ]; then EXTRA="--m6-reward-mult 1.0"; fi
    python3 train_$alg.py --device cpu --seed $s --refined-reward --refined-m2-boost 0.05 $EXTRA >/dev/null 2>&1
    python3 eval_$alg.py  --device cpu --refined-reward --refined-m2-boost 0.05 >/dev/null 2>&1
    python3 - <<PY
import json,shutil,os
os.makedirs("resultados_uniforme",exist_ok=True)
d=json.load(open("models/$alg/eval_metrics.json"))
d["alg"]="$alg"; d["seed"]=$s
json.dump(d,open(f"resultados_uniforme/${alg}_seed$s.json","w"))
PY
  done
done

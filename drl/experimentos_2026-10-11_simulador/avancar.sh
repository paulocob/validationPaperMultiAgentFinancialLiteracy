#!/bin/bash
# Avança uma rodada do planejador LLM e gera a visão em texto da seguinte.
# Uso: bash avancar.sh <pasta> <k>   (k = rodada cujas decisões serão aplicadas)
set -e
d=$1; k=$(printf "%02d" $2); n=$(printf "%02d" $(($2+1)))
python planejador_rodadas.py step --dir $d/ --decisoes $d/decisoes_$k.json 2>&1 | tail -1
if [ -f $d/FIM ]; then echo "FIM em $d"; else python rodada_para_texto.py $d $(($2+1)); fi

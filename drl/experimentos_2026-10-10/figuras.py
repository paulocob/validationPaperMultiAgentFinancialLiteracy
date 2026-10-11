"""Regenera as três figuras do capítulo de DRL a partir de resultados.json (E8).

  fig_teto_metrica.pdf     recompensa associada vs teto, por política (5-fold)
  fig_fronteira_lambda.pdf Top-1 e recompensa associada vs λ_reg (val e teste)
  fig_dma_segmento.pdf     Top-1 por segmento, DQN λ* (5-fold OOF)

Largura de texto do documento ≈ 15 cm (5,9 in). Rótulos em português, vírgula
decimal, fonte serifada ≥ 8 pt, sem título dentro da figura. Os PDFs antigos
são copiados para figures/_antigas_2026-10-10/ antes de serem sobrescritos.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

AQUI = Path(__file__).resolve().parent
# No repositório de validação, as figuras vão para ./figuras/ desta pasta.
# A variável de ambiente FIG_DIR permite outro destino (na tese, a pasta figures/ do documento).
FIG_DIR = Path(os.environ.get("FIG_DIR", str(AQUI / "figuras")))
BACKUP = FIG_DIR / "_antigas_2026-10-10"
NOMES = ["fig_teto_metrica.pdf", "fig_fronteira_lambda.pdf", "fig_dma_segmento.pdf"]

# Paleta categórica validada (dataviz/validate_palette.js, modo claro): azul e
# laranja em ordem fixa; cinza neutro para o teto (referência, não uma série).
AZUL, LARANJA, CINZA, TEXTO = "#2a78d6", "#eb6834", "#9a9a94", "#222222"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#555555",
    "pdf.fonttype": 42,
    # A matemática do mathtext usa DejaVu Sans por padrão; sem isto, subscritos
    # como o "reg" de λ_reg saíam sem serifa no meio de rótulos em Times.
    "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
})
VIRGULA = FuncFormatter(lambda v, _: ("%g" % round(v, 3)).replace(".", ","))
POL = [("seq", "Regra\nsequencial"), ("bc", "BC"), ("dqn0", "DQN\nλ = 0"), ("dqn_lstar", "DQN\nλ*"), ("dqn_l1se", "DQN\nλ$_{1EP}$"),
       ("cql", "CQL"), ("ppo", "PPO\noffline")]


def _backup():
    BACKUP.mkdir(parents=True, exist_ok=True)
    for n in NOMES:
        src = FIG_DIR / n
        if src.exists() and not (BACKUP / n).exists():
            shutil.copy2(src, BACKUP / n)


def fig_teto(R):
    cv = R["cv"]["politicas"]
    x = np.arange(len(POL))
    rec = [cv[p]["reward_assoc"][0] for p, _ in POL]
    rec_sd = [cv[p]["reward_assoc"][1] for p, _ in POL]
    tet = [cv[p]["teto"][0] for p, _ in POL]
    tet_sd = [cv[p]["teto"][1] for p, _ in POL]
    fig, ax = plt.subplots(figsize=(5.9, 3.1))
    w = 0.36
    b1 = ax.bar(x - w / 2, rec, w, color=AZUL, label="recompensa associada (observada)", yerr=rec_sd,
                error_kw=dict(ecolor=TEXTO, elinewidth=0.8, capsize=2))
    b2 = ax.bar(x + w / 2, tet, w, color=CINZA, label="teto para a mesma fração de acertos", yerr=tet_sd,
                error_kw=dict(ecolor=TEXTO, elinewidth=0.8, capsize=2))
    for bars, vals in ((b1, rec), (b2, tet)):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, 0.02, ("%.3f" % v).replace(".", ","), ha="center", va="bottom",
                    fontsize=7.5, color="white", rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in POL])
    ax.set_ylabel("recompensa média nos acertos")
    ax.set_ylim(0, 0.9)
    ax.yaxis.set_major_formatter(VIRGULA)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_teto_metrica.pdf")
    plt.close(fig)


def fig_fronteira(R):
    g = R["split_fixo"]["grade_lambda"]
    lams = [float(k) for k in g]
    x = np.arange(len(lams))
    lstar = R["split_fixo"]["lambda_star"]
    fig, axes = plt.subplots(1, 2, figsize=(5.9, 2.9), sharex=True)
    for ax, (kv, kt, ylab) in zip(axes, [("val_top1", "test_top1", "Top-1"),
                                         ("val_reward_assoc", "test_reward_assoc", "recompensa associada")]):
        for key, cor, lab, mk in ((kv, LARANJA, "validação", "s"), (kt, AZUL, "teste", "o")):
            m = [g[k][key][0] for k in g]
            sd = [g[k][key][1] for k in g]
            ax.errorbar(x, m, yerr=sd, color=cor, marker=mk, ms=4, lw=1.4, capsize=2, elinewidth=0.8, label=lab)
        # λ* (Top-1 máximo) e λ_1EP (regra de um erro-padrão com EP binomial,
        # o mesmo critério do 5-fold); se coincidirem, uma linha só.
        marcas = {lstar: "λ*"}
        l1 = R["split_fixo"]["lambda_1se_binomial"]
        marcas[l1] = (marcas[l1] + " = " if l1 in marcas else "") + "λ$_{1EP}$"
        for j, (lam_sel, rot) in enumerate(marcas.items()):
            i = lams.index(lam_sel)
            ax.axvline(i, color=CINZA, lw=0.9, ls="--" if j == 0 else ":")
            ax.text(i, ax.get_ylim()[1], " " + rot, color=TEXTO, va="top", ha="left", fontsize=8.5)
        ax.set_ylabel(ylab)
        ax.set_xticks(x)
        ax.set_xticklabels([("%g" % l).replace(".", ",") for l in lams], rotation=45)
        ax.set_xlabel("λ$_{reg}$")
        ax.yaxis.set_major_formatter(VIRGULA)
    axes[0].legend(frameon=False, loc="best")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(FIG_DIR / "fig_fronteira_lambda.pdf")
    plt.close(fig)


def fig_segmento(R):
    seg = R["cv"]["top1_por_segmento"]
    ordem = ["iniciante", "intermediario", "avancado"]
    rot = {"iniciante": "iniciante", "intermediario": "intermediário", "avancado": "avançado"}
    vals = [seg[s]["top1"] for s in ordem]
    sds = [seg[s]["dp_entre_folds"] for s in ordem]
    ns = [seg[s]["n"] for s in ordem]
    geral = R["cv"]["politicas"]["dqn_lstar"]["pooled_oof"]["top1"]
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    bars = ax.bar(np.arange(3), vals, 0.55, color=AZUL, yerr=sds, error_kw=dict(ecolor=TEXTO, elinewidth=0.8, capsize=3))
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v - 0.05, ("%.3f" % v).replace(".", ","),
                ha="center", va="top", fontsize=8.5, color="white")
    ax.axhline(geral, color=CINZA, ls="--", lw=0.9)
    ax.text(-0.42, geral + 0.012, "geral " + ("%.3f" % geral).replace(".", ","), ha="left", va="bottom", fontsize=8, color=TEXTO)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels([f"{rot[s]}\n(n = {n})" for s, n in zip(ordem, ns)])
    ax.set_ylabel("Top-1 (DQN λ*, 5-fold por perfil)")
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(VIRGULA)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_dma_segmento.pdf")
    plt.close(fig)


def gerar(R):
    _backup()
    fig_teto(R)
    fig_fronteira(R)
    fig_segmento(R)
    print(f"[figuras] regeneradas em {FIG_DIR} (backup em {BACKUP})")


if __name__ == "__main__":
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else AQUI / "resultados.json"
    gerar(json.loads(p.read_text(encoding="utf-8")))

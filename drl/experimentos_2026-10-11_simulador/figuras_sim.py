"""Figuras da 7ª rodada (A4), geradas de resultados.json.

  fig_sim_hake_heterogeneidade.pdf  Hake imediato por política × h, IC 95 %
  fig_sim_abandono.pdf              abandono por política × h, IC 95 %
  fig_sim_ope_validacao.pdf         valor verdadeiro × estimado (SNIS, DR), por h

Largura 15 cm (5,9 in), rótulos em português, vírgula decimal, fonte ≥ 8 pt, sem
título interno. Nomes NOVOS: nenhuma figura existente é sobrescrita. Na fumaça
(--quick) as figuras vão para ./figuras_quick/, não para a pasta do documento.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

AQUI = Path(__file__).resolve().parent
# No repositório de validação, as figuras vão para ./figuras/ desta pasta.
# A variável de ambiente FIG_DIR permite outro destino (na tese, a pasta figures/ do documento).
FIG_DIR = Path(os.environ.get("FIG_DIR", str(AQUI / "figuras")))
LARGURA = 15 / 2.54

# h é ordinal: três tons do mesmo azul da paleta usada nas figuras de 10/10.
COR_H = {0.0: "#9cc3ee", 0.5: "#2a78d6", 1.0: "#0d3b78"}
MARCA_H = {0.0: "o", 0.5: "s", 1.0: "D"}
LARANJA, CINZA, TEXTO = "#eb6834", "#9a9a94", "#222222"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#555555", "pdf.fonttype": 42,
})
VIRG = FuncFormatter(lambda v, _: ("%g" % round(v, 3)).replace(".", ","))
CURTO = {
    "curriculo": "Currículo fixo", "aleatoria": "Aleatória (pré-req.)", "linucb": "LinUCB", "ts": "Thompson linear",
    "dqn_off_1ep": "DQN offline (λ Top-1/1EP)", "dqn_off_snis": "DQN offline (λ SNIS)", "dqn_on": "DQN online",
    "llm_zero": "LLM zero-shot", "llm_few": "LLM few-shot", "guloso_kt": "Guloso pelo rastreador", "oraculo": "Oráculo míope (teto)",
}
SIGLA = {"curriculo": "C", "linucb": "U", "ts": "T", "dqn_off_1ep": "D1", "dqn_off_snis": "DS", "dqn_on": "DO",
         "llm_zero": "L0", "llm_few": "L3"}
HS = [0.0, 0.5, 1.0]


def _hlabel(h):
    return "h = " + ("%g" % h).replace(".", ",")


def _politicas(R):
    ordem = ["oraculo", "guloso_kt", "dqn_on", "dqn_off_snis", "dqn_off_1ep", "ts", "linucb", "llm_few", "llm_zero",
             "aleatoria", "curriculo"]
    return [p for p in ordem if all(f"{p}|{h}" in R["tabela"] for h in HS)]


def _dotplot(R, chave, xlabel, destino, pct=False):
    pols = _politicas(R)
    fig, ax = plt.subplots(figsize=(LARGURA, 0.36 * len(pols) + 0.9))
    y = np.arange(len(pols))
    desloc = {0.0: -0.22, 0.5: 0.0, 1.0: 0.22}
    esc = 100.0 if pct else 1.0
    for h in HS:
        m = np.array([R["tabela"][f"{p}|{h}"][chave] for p in pols]) * esc
        ic = np.array([R["tabela"][f"{p}|{h}"][chave + "_ic"] for p in pols]) * esc
        ax.errorbar(m, y + desloc[h], xerr=[m - ic[:, 0], ic[:, 1] - m], fmt=MARCA_H[h], color=COR_H[h], ms=4,
                    elinewidth=0.9, capsize=1.5, label=_hlabel(h))
    # Referência: currículo em cada h (linha tracejada da cor do h).
    for h in HS:
        ax.axvline(R["tabela"][f"curriculo|{h}"][chave] * esc, color=COR_H[h], lw=0.7, ls="--", alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([CURTO[p] for p in pols])
    ax.set_xlabel(xlabel)
    ax.xaxis.set_major_formatter(VIRG)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    fig.tight_layout()
    Path(destino).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino)
    plt.close(fig)


def fig_ope(R, destino):
    fig, axes = plt.subplots(1, 3, figsize=(LARGURA, 2.8), sharex=False, sharey=False)
    for ax, h in zip(axes, HS):
        linhas = R["ope"][str(h)]["politicas"]
        v = np.array([linhas[n]["verdadeiro_rec_passo"] for n in linhas])
        s = np.array([linhas[n]["snis"] for n in linhas])
        d = np.array([linhas[n]["dr"] for n in linhas])
        lo, hi = min(v.min(), s.min(), d.min()), max(v.max(), s.max(), d.max())
        pad = 0.08 * (hi - lo + 1e-6)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=CINZA, lw=0.8, ls="--")
        ax.scatter(v, s, marker="o", s=16, color=COR_H[0.5], label="SNIS", zorder=3)
        ax.scatter(v, d, marker="^", s=16, color=LARANJA, label="DR", zorder=3)
        for n, xv, yv in zip(linhas, v, s):
            ax.annotate(SIGLA.get(n, n), (xv, yv), textcoords="offset points", xytext=(3, 2), fontsize=7, color=TEXTO)
        c = R["ope"][str(h)]["concordancia"]
        ax.set_title(f"{_hlabel(h)}  (τ SNIS = {c['snis']['kendall_vs_rec_passo']:.2f}; DR = "
                     f"{c['dr']['kendall_vs_rec_passo']:.2f})".replace(".", ","), fontsize=8)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.xaxis.set_major_formatter(VIRG)
        ax.yaxis.set_major_formatter(VIRG)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_xlabel("verdadeiro (recompensa/passo)")
    axes[0].set_ylabel("estimado nos logs")
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(w_pad=0.8, rect=(0, 0, 1, 0.92))
    fig.savefig(destino)
    plt.close(fig)


def fig_sensibilidade(R, destino):
    """Δg vs currículo (h = 1) nas 27 combinações; um painel por escala de dispersão,
    eixo x = (multiplicador, limiar) de pré-requisito."""
    S = R["sensibilidade"]
    G = S["grade"]
    series = [("oraculo", "Oráculo míope (teto)", "#0d3b78", "D"), ("guloso_kt", "Guloso pelo rastreador", "#2a78d6", "s"),
              ("dqn_on", "DQN online", LARANJA, "o"), ("linucb", "LinUCB", CINZA, "^")]
    fig, axes = plt.subplots(1, 3, figsize=(LARGURA, 3.0), sharey=True)
    pares = [(m, l) for m in G["mult_prereq"] for l in G["limiar_prereq"]]
    x = np.arange(len(pares))
    for ax, esc in zip(axes, G["escala_dispersao"]):
        lin = {(c["mult"], c["limiar"]): c for c in S["combinacoes"] if c["escala"] == esc}
        for j, (pol, rot, cor, mk) in enumerate(series):
            y = np.array([lin[pr]["politicas"][pol]["dg"] for pr in pares])
            ic = np.array([lin[pr]["politicas"][pol]["dg_ic"] for pr in pares])
            ax.errorbar(x + (j - 1.5) * 0.12, y, yerr=[y - ic[:, 0], ic[:, 1] - y], fmt=mk, color=cor, ms=3.2,
                        elinewidth=0.7, capsize=0, label=rot)
        ax.axhline(0, color=TEXTO, lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{('%g' % m).replace('.', ',')}/{('%g' % l).replace('.', ',')}" for m, l in pares],
                           rotation=90, fontsize=7.5)
        ax.set_title("escala de dispersão " + ("%g" % esc).replace(".", ","), fontsize=8.5)
        ax.yaxis.set_major_formatter(VIRG)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Δg vs currículo (h = 1)")
    fig.supxlabel("multiplicador / limiar de pré-requisito", fontsize=8.5)
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=0.6)
    fig.savefig(destino)
    plt.close(fig)


def gerar(R, sufixo: str = ""):
    destino = FIG_DIR if not sufixo else AQUI / "figuras_quick"
    destino.mkdir(parents=True, exist_ok=True)
    _dotplot(R, "hake", "ganho normalizado de Hake no pós-teste imediato (média, IC 95 %)",
             destino / "fig_sim_hake_heterogeneidade.pdf")
    _dotplot(R, "abandono", "abandono (% dos aprendizes, IC 95 %)", destino / "fig_sim_abandono.pdf", pct=True)
    fig_ope(R, destino / "fig_sim_ope_validacao.pdf")
    if "sensibilidade" in R and not R["sensibilidade"].get("quick"):
        fig_sensibilidade(R, destino / "fig_sim_sensibilidade.pdf")
    print(f"[figuras] em {destino}")


if __name__ == "__main__":
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else AQUI / "resultados.json"
    gerar(json.loads(p.read_text(encoding="utf-8")))

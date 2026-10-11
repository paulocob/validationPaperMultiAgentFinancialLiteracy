"""
Estatisticas descritivas e testes de cobertura/balanceamento do FinEduBench-Luso.

Uso:
    python3 -m dataset.stats
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median, stdev

from dataset.schema import QAPair, load_all_countries


TOPICS = ["budget", "savings", "debt", "investment", "credit", "cultural"]
DIFFICULTIES = ["basic", "intermediate", "advanced"]
ALL_MODALITIES = ["text", "image", "audio"]


def _len_chars(s: str | None) -> int:
    return len(s) if s else 0


def _len_words(s: str | None) -> int:
    return len(s.split()) if s else 0


def _chi_square_uniform(observed: list[int]) -> tuple[float, int]:
    """Teste qui-quadrado contra distribuicao uniforme. Retorna (chi2, df)."""
    n = sum(observed)
    k = len(observed)
    expected = n / k
    chi2 = sum((o - expected) ** 2 / expected for o in observed)
    return chi2, k - 1


def _chi_square_critical_5pct(df: int) -> float:
    """Valores criticos qui-quadrado a 5% para df pequenos."""
    table = {1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488, 5: 11.070,
             6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919, 10: 18.307,
             12: 21.026, 15: 24.996, 20: 31.410}
    return table.get(df, float("inf"))


def print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(f" {title}")
    print("=" * 72)


def stats_per_country(pairs: list[QAPair], country: str) -> dict:
    n = len(pairs)
    topic_counts = Counter(qa.topic for qa in pairs)
    diff_counts = Counter(qa.difficulty for qa in pairs)

    # Modalidades: contamos pares por combinacao e por modalidade individual
    mod_combo = Counter("+".join(sorted(qa.modalities)) for qa in pairs)
    mod_individual = Counter()
    for qa in pairs:
        for m in qa.modalities:
            mod_individual[m] += 1

    # Comprimentos (tokens aproximados via word split)
    q_words = [_len_words(qa.question) for qa in pairs]
    a_words = [_len_words(qa.answer) for qa in pairs]
    img_desc_words = [_len_words(qa.image_description) for qa in pairs if qa.image_description]
    audio_words = [_len_words(qa.audio_transcript) for qa in pairs if qa.audio_transcript]
    cultural_words = [_len_words(qa.cultural_notes) for qa in pairs if qa.cultural_notes]

    # Fontes
    n_with_sources = sum(1 for qa in pairs if qa.sources)
    n_sources_total = sum(len(qa.sources or []) for qa in pairs)

    return {
        "country": country,
        "n": n,
        "topics": topic_counts,
        "difficulty": diff_counts,
        "mod_combo": mod_combo,
        "mod_individual": mod_individual,
        "q_words": q_words,
        "a_words": a_words,
        "img_desc_words": img_desc_words,
        "audio_words": audio_words,
        "cultural_words": cultural_words,
        "n_with_sources": n_with_sources,
        "n_sources_total": n_sources_total,
    }


def format_distribution(counts: Counter, expected_keys: list[str], total: int) -> str:
    parts = []
    for k in expected_keys:
        n = counts.get(k, 0)
        pct = 100.0 * n / total if total else 0.0
        parts.append(f"{k}={n} ({pct:.1f}%)")
    return ", ".join(parts)


def print_summary(stats: dict) -> None:
    c = stats["country"]
    n = stats["n"]
    print(f"\n[{c}] n = {n}")

    print(f"  Topicos:     {format_distribution(stats['topics'], TOPICS, n)}")
    chi2, df = _chi_square_uniform([stats["topics"].get(t, 0) for t in TOPICS])
    crit = _chi_square_critical_5pct(df)
    verdict = "OK (p>0.05)" if chi2 < crit else "desbalanceado"
    print(f"               chi2={chi2:.2f} (df={df}, critico5%={crit:.2f}) -> {verdict}")

    print(f"  Dificuldade: {format_distribution(stats['difficulty'], DIFFICULTIES, n)}")

    print(f"  Modalidade (combinacoes): {dict(sorted(stats['mod_combo'].items()))}")
    print(f"  Modalidade (individual):  {dict(sorted(stats['mod_individual'].items()))}")
    mm_text = stats["mod_individual"].get("text", 0)
    mm_img = stats["mod_individual"].get("image", 0)
    mm_audio = stats["mod_individual"].get("audio", 0)
    print(f"    -> cobertura texto={100*mm_text/n:.1f}%, imagem={100*mm_img/n:.1f}%, audio={100*mm_audio/n:.1f}%")

    def _describe(label: str, xs: list[int]) -> str:
        if not xs:
            return f"{label}: (vazio)"
        sd = stdev(xs) if len(xs) > 1 else 0.0
        return (f"{label}: n={len(xs)}, mean={mean(xs):.1f}, "
                f"median={median(xs):.1f}, sd={sd:.1f}, min={min(xs)}, max={max(xs)}")

    print(f"  Tamanhos (palavras):")
    print(f"    {_describe('pergunta        ', stats['q_words'])}")
    print(f"    {_describe('resposta        ', stats['a_words'])}")
    print(f"    {_describe('image_desc      ', stats['img_desc_words'])}")
    print(f"    {_describe('audio_transcript', stats['audio_words'])}")
    print(f"    {_describe('cultural_notes  ', stats['cultural_words'])}")

    src_cov = 100.0 * stats["n_with_sources"] / n
    avg_src = stats["n_sources_total"] / n
    print(f"  Fontes: {stats['n_with_sources']}/{n} com fontes ({src_cov:.1f}%), "
          f"media {avg_src:.2f} por QA")


def cross_country_comparison(all_stats: list[dict]) -> None:
    print_header("Comparacao transcultural")

    # Qui-quadrado: dificuldade x pais
    print("\nQui-quadrado dificuldade x pais (independencia):")
    countries = [s["country"] for s in all_stats]
    observed_matrix = []
    for s in all_stats:
        row = [s["difficulty"].get(d, 0) for d in DIFFICULTIES]
        observed_matrix.append(row)

    row_totals = [sum(row) for row in observed_matrix]
    col_totals = [sum(col) for col in zip(*observed_matrix)]
    grand = sum(row_totals)
    chi2 = 0.0
    for i in range(len(observed_matrix)):
        for j in range(len(DIFFICULTIES)):
            expected = row_totals[i] * col_totals[j] / grand
            obs = observed_matrix[i][j]
            if expected > 0:
                chi2 += (obs - expected) ** 2 / expected
    df = (len(observed_matrix) - 1) * (len(DIFFICULTIES) - 1)
    crit = _chi_square_critical_5pct(df)
    verdict = "OK (independente)" if chi2 < crit else "dependente"
    print(f"  chi2 = {chi2:.2f}, df = {df}, critico5% = {crit:.2f} -> {verdict}")

    # Qui-quadrado: topico x pais
    print("\nQui-quadrado topico x pais (independencia):")
    observed_matrix = [[s["topics"].get(t, 0) for t in TOPICS] for s in all_stats]
    row_totals = [sum(row) for row in observed_matrix]
    col_totals = [sum(col) for col in zip(*observed_matrix)]
    grand = sum(row_totals)
    chi2 = 0.0
    for i in range(len(observed_matrix)):
        for j in range(len(TOPICS)):
            expected = row_totals[i] * col_totals[j] / grand
            if expected > 0:
                chi2 += (observed_matrix[i][j] - expected) ** 2 / expected
    df = (len(observed_matrix) - 1) * (len(TOPICS) - 1)
    crit = _chi_square_critical_5pct(df)
    verdict = "OK (independente)" if chi2 < crit else "dependente"
    print(f"  chi2 = {chi2:.2f}, df = {df}, critico5% = {crit:.2f} -> {verdict}")

    # Comprimento medio das respostas por pais
    print("\nComprimento medio (palavras) da resposta por pais:")
    for s in all_stats:
        print(f"  {s['country']}: {mean(s['a_words']):.1f} +/- {stdev(s['a_words']):.1f}")

    # Cobertura de modalidades por pais
    print("\nCobertura de modalidade (% de QAs usando cada modalidade):")
    print(f"  {'pais':6s} {'texto':>10s} {'imagem':>10s} {'audio':>10s}")
    for s in all_stats:
        n = s["n"]
        mi = s["mod_individual"]
        print(f"  {s['country']:6s} "
              f"{100*mi.get('text', 0)/n:>9.1f}% "
              f"{100*mi.get('image', 0)/n:>9.1f}% "
              f"{100*mi.get('audio', 0)/n:>9.1f}%")


def check_id_uniqueness(country_data: dict[str, list[QAPair]]) -> None:
    print_header("Verificacao de unicidade de IDs")
    total_ids = []
    for cc, pairs in country_data.items():
        ids = [qa.id for qa in pairs]
        total_ids.extend(ids)
        dup = [k for k, v in Counter(ids).items() if v > 1]
        status = "OK" if not dup else f"DUPLICADOS: {dup}"
        print(f"  {cc}: {len(ids)} IDs, {len(set(ids))} unicos -> {status}")
    all_dup = [k for k, v in Counter(total_ids).items() if v > 1]
    print(f"  GLOBAL: {len(total_ids)} IDs, {len(set(total_ids))} unicos "
          f"-> {'OK' if not all_dup else f'DUPLICADOS: {all_dup}'}")


def main() -> int:
    base = Path(__file__).resolve().parent
    all_pairs = load_all_countries(base)
    if not all_pairs:
        print("Nenhum dado encontrado. Rode primeiro: python3 -m dataset.generator")
        return 1

    country_data: dict[str, list[QAPair]] = {"BR": [], "PT": [], "AO": []}
    for qa in all_pairs:
        if qa.country in country_data:
            country_data[qa.country].append(qa)

    print_header("FinEduBench-Luso: estatisticas descritivas")
    all_stats = []
    for cc in ("BR", "PT", "AO"):
        if cc not in country_data:
            continue
        s = stats_per_country(country_data[cc], cc)
        all_stats.append(s)
        print_summary(s)

    cross_country_comparison(all_stats)
    check_id_uniqueness(country_data)

    total = sum(s["n"] for s in all_stats)
    print_header(f"Total global: {total} QA pairs")
    return 0


if __name__ == "__main__":
    sys.exit(main())

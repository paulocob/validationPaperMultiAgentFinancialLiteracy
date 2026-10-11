"""
Script de avaliacao do FinEduBench-Luso.

Carrega predicoes e dados de referencia, executa todas as metricas,
e gera relatorios por pais, topico e dificuldade.

Uso:
    python -m evaluation.run_evaluation --predictions <caminho_predicoes>
    python -m evaluation.run_evaluation --predictions baselines/results/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

# Adiciona o diretorio raiz ao path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset.schema import load_all_countries
from evaluation.metrics import BenchmarkEvaluator

console = Console()


def load_predictions(path: str | Path) -> list[dict]:
    """
    Carrega predicoes de arquivo JSONL ou diretorio com arquivos JSONL.

    Formato esperado por linha: {"id": "...", "predicted_answer": "...", "predicted_modalities": [...]}
    """
    path = Path(path)
    predictions: list[dict] = []

    if path.is_dir():
        jsonl_files = sorted(path.glob("*.jsonl"))
        if not jsonl_files:
            console.print(f"[red]Nenhum arquivo .jsonl encontrado em {path}[/red]")
            return predictions
        for f in jsonl_files:
            console.print(f"  Carregando predicoes de: {f.name}")
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        predictions.append(json.loads(line))
    elif path.is_file():
        console.print(f"  Carregando predicoes de: {path.name}")
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    predictions.append(json.loads(line))
    else:
        console.print(f"[red]Caminho nao encontrado: {path}[/red]")

    return predictions


def print_rich_table(title: str, df: pd.DataFrame) -> None:
    """Imprime um DataFrame como tabela formatada no terminal."""
    table = Table(title=title, show_lines=True)
    table.add_column("Grupo", style="bold cyan")
    for col in df.columns:
        table.add_column(col, justify="right")

    for idx, row in df.iterrows():
        values = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in row.values]
        table.add_row(str(idx), *values)

    console.print(table)


def main() -> None:
    """Executa a avaliacao completa do benchmark."""
    parser = argparse.ArgumentParser(
        description="Avaliacao do FinEduBench-Luso"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Caminho para arquivo JSONL de predicoes ou diretorio com arquivos JSONL",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(_PROJECT_ROOT / "dataset"),
        help="Diretorio base do dataset (default: dataset/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_PROJECT_ROOT / "evaluation" / "results"),
        help="Diretorio para salvar resultados",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Carregar dados ---
    console.print("\n[bold green]1. Carregando dados de referencia...[/bold green]")
    gold_qa_pairs = load_all_countries(args.dataset_dir)
    if not gold_qa_pairs:
        console.print("[red]Nenhum dado de referencia encontrado. Execute primeiro:[/red]")
        console.print("  python -m dataset.generator")
        sys.exit(1)
    console.print(f"  {len(gold_qa_pairs)} QA pairs de referencia carregados.")

    console.print("\n[bold green]2. Carregando predicoes...[/bold green]")
    predictions = load_predictions(args.predictions)
    if not predictions:
        console.print("[red]Nenhuma predicao encontrada.[/red]")
        sys.exit(1)
    console.print(f"  {len(predictions)} predicoes carregadas.")

    # --- 2. Avaliar ---
    console.print("\n[bold green]3. Executando avaliacao...[/bold green]")
    evaluator = BenchmarkEvaluator()
    gold_dicts = [qa.to_dict() for qa in gold_qa_pairs]
    results_df = evaluator.evaluate_batch(predictions, gold_dicts)
    console.print(f"  {len(results_df)} itens avaliados.")

    if results_df.empty:
        console.print("[red]Nenhum resultado gerado. Verifique os IDs das predicoes.[/red]")
        sys.exit(1)

    # --- 3. Relatorios ---
    console.print("\n[bold green]4. Gerando relatorios...[/bold green]")
    summary = evaluator.summary()

    # Por pais
    by_country = summary["by_country"]
    mean_cols_country = [c for c in by_country.columns if c.endswith("_mean")]
    if mean_cols_country:
        display_country = by_country[mean_cols_country].copy()
        display_country.columns = [c.replace("_mean", "") for c in mean_cols_country]
        print_rich_table("Resultados por Pais", display_country)

    # Por topico
    by_topic = summary["by_topic"]
    mean_cols_topic = [c for c in by_topic.columns if c.endswith("_mean")]
    if mean_cols_topic:
        display_topic = by_topic[mean_cols_topic].copy()
        display_topic.columns = [c.replace("_mean", "") for c in mean_cols_topic]
        print_rich_table("Resultados por Topico", display_topic)

    # Por dificuldade
    by_diff = summary["by_difficulty"]
    mean_cols_diff = [c for c in by_diff.columns if c.endswith("_mean")]
    if mean_cols_diff:
        display_diff = by_diff[mean_cols_diff].copy()
        display_diff.columns = [c.replace("_mean", "") for c in mean_cols_diff]
        print_rich_table("Resultados por Dificuldade", display_diff)

    # Overall
    overall = summary["overall"]
    if not overall.empty:
        console.print("\n[bold]Estatisticas Gerais:[/bold]")
        console.print(overall.to_string())

    # --- 4. Salvar resultados ---
    console.print(f"\n[bold green]5. Salvando resultados em {output_dir}/[/bold green]")

    # CSV completo
    results_df.to_csv(output_dir / "results_full.csv", index=False)
    console.print("  results_full.csv salvo.")

    # CSVs agregados
    if not by_country.empty:
        by_country.to_csv(output_dir / "results_by_country.csv")
    if not by_topic.empty:
        by_topic.to_csv(output_dir / "results_by_topic.csv")
    if not by_diff.empty:
        by_diff.to_csv(output_dir / "results_by_difficulty.csv")

    # Tabela LaTeX
    latex_country = evaluator.to_latex(by="country")
    latex_topic = evaluator.to_latex(by="topic")
    latex_diff = evaluator.to_latex(by="difficulty")

    latex_output = (
        "% Tabelas geradas automaticamente pelo FinEduBench-Luso\n"
        "% Data de geracao: ver metadata do arquivo\n\n"
        f"% --- Por Pais ---\n{latex_country}\n\n"
        f"% --- Por Topico ---\n{latex_topic}\n\n"
        f"% --- Por Dificuldade ---\n{latex_diff}\n"
    )
    latex_path = output_dir / "results_tables.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_output)
    console.print(f"  Tabelas LaTeX salvas em {latex_path.name}")

    console.print("\n[bold green]Avaliacao concluida![/bold green]\n")


if __name__ == "__main__":
    main()

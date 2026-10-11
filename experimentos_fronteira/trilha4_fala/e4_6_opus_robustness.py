"""
E4.6 — Robustez do classificador Wav2Vec2 ao domínio WhatsApp (Opus).

Hipótese: modelo treinado em WAV 16kHz limpo pode degradar 10-30 pp quando
recebe áudio Opus comprimido (formato real do WhatsApp).

Metodologia:
  1. Carregar as 737 amostras de teste.
  2. Sintetizar degradação Opus via ffmpeg.
  3. Reavaliar modelo sobre os mesmos áudios degradados.
  4. Reportar diferença de acurácia em pp.

Saída: resultados/e4_6_opus_robustness.json
Tempo estimado em M4 Pro: ~1h (CPU-bound para conversão + inferência).

Requer ffmpeg: brew install ffmpeg
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.logging import get_logger
from common.hardware import print_hardware_summary

ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT.parent
VARIANTES_DIR = REPO_ROOT / "sbrt2026-lusophone-variants"
RESULTS_DIR = ROOT / "resultados"
RESULTS_DIR.mkdir(exist_ok=True)


# Configurações de degradação Opus (do mais leve ao mais agressivo)
OPUS_BITRATES = [
    ("opus_32k", 32),   # WhatsApp típico com boa conexão
    ("opus_16k", 16),   # WhatsApp em conexão moderada
    ("opus_8k",  8),    # WhatsApp em conexão ruim
]


def check_ffmpeg():
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg não encontrado. Instale com: brew install ffmpeg")


def degrade_wav_to_opus(input_wav: Path, output_wav: Path, bitrate_kbps: int):
    """
    Round-trip WAV -> Opus -> WAV para simular degradação do WhatsApp.
    O áudio final permanece como WAV 16kHz mono (compatível com Wav2Vec2),
    mas carrega as perdas da compressão Opus.
    """
    intermediate_opus = output_wav.with_suffix(".opus")
    # WAV -> Opus
    subprocess.check_call([
        "ffmpeg", "-y", "-i", str(input_wav),
        "-c:a", "libopus", "-b:a", f"{bitrate_kbps}k",
        "-ar", "16000", "-ac", "1",
        str(intermediate_opus),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Opus -> WAV
    subprocess.check_call([
        "ffmpeg", "-y", "-i", str(intermediate_opus),
        "-ar", "16000", "-ac", "1",
        str(output_wav),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    intermediate_opus.unlink()


def run_robustness(dry_run: bool = False):
    logger = get_logger("e4_6_opus", output_dir=RESULTS_DIR / "logs")
    logger.info("=== E4.6 Robustez a Opus (WhatsApp) ===")

    if not dry_run:
        check_ffmpeg()

    # Localizar arquivos WAV de teste
    # NOTA: estrutura do dataset varia; ajustar conforme organização.
    test_wav_dir = VARIANTES_DIR / "data" / "test" / "wav"
    if not test_wav_dir.exists():
        logger.warning(f"Diretório {test_wav_dir} não encontrado. "
                       "Ajustar path conforme organização do seu dataset.")
        return {"error": "test_wav_dir não encontrado"}

    wav_files = sorted(test_wav_dir.rglob("*.wav"))
    logger.info(f"Arquivos de teste: {len(wav_files)}")

    if dry_run:
        logger.info("[DRY RUN] Amostra dos primeiros 3 arquivos:")
        for w in wav_files[:3]:
            logger.info(f"  {w}")
        return {"dry_run": True, "n_wavs": len(wav_files)}

    results = {}
    for name, bitrate in OPUS_BITRATES:
        logger.info(f"→ Degradando para {name} ({bitrate} kbps)...")
        degraded_dir = VARIANTES_DIR / "data" / "test" / name
        degraded_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        for i, wav in enumerate(wav_files):
            out_wav = degraded_dir / wav.relative_to(test_wav_dir)
            out_wav.parent.mkdir(parents=True, exist_ok=True)
            try:
                degrade_wav_to_opus(wav, out_wav, bitrate)
            except Exception as e:
                logger.error(f"  falhou em {wav}: {e}")
        logger.info(f"  {len(wav_files)} arquivos degradados em {time.time()-t0:.1f}s")

        # Rodar avaliação do modelo sobre áudios degradados
        evaluate_script = VARIANTES_DIR / "src" / "evaluate.py"
        eval_output = RESULTS_DIR / "models" / f"eval_opus_{name}"
        eval_output.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, str(evaluate_script),
            "--test-dir", str(degraded_dir),
            "--output", str(eval_output),
        ]
        logger.info(f"  Avaliando modelo em {name}...")
        try:
            subprocess.check_call(cmd, cwd=VARIANTES_DIR)
            acc_file = eval_output / "accuracy.json"
            if acc_file.exists():
                results[name] = json.loads(acc_file.read_text())
        except Exception as e:
            logger.error(f"  avaliação falhou: {e}")
            results[name] = {"error": str(e)}

    # Comparar com baseline WAV limpo (91.5%)
    baseline_acc = 0.9145
    summary = {
        "exp_id": "E4.6",
        "baseline_wav_accuracy": baseline_acc,
        "degraded": results,
        "delta_pp": {
            name: round((r.get("accuracy", 0) - baseline_acc) * 100, 2)
            for name, r in results.items()
            if isinstance(r, dict) and r.get("accuracy")
        },
    }
    out = RESULTS_DIR / "e4_6_opus_robustness.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info(f"✓ Salvo em {out}")
    for name, delta in summary.get("delta_pp", {}).items():
        logger.info(f"  {name}: Δ = {delta:+.2f} pp")
    return summary


def main():
    parser = argparse.ArgumentParser(description="E4.6 — Robustez a Opus")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print_hardware_summary()
    run_robustness(dry_run=args.dry_run)


if __name__ == "__main__":
    main()

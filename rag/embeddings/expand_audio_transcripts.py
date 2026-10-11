"""
Expande audio_transcript para 100% dos QA pairs usando GPT-4o-mini.

Gera transcricoes coloquiais/orais para os 942 QA pairs que nao possuem
audio_transcript, mantendo os 57 existentes intactos.

Uso:
    OPENAI_API_KEY=sk-... python3 embeddings/expand_audio_transcripts.py
    OPENAI_API_KEY=sk-... python3 embeddings/expand_audio_transcripts.py --sample 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset.schema import load_jsonl, save_jsonl, QAPair

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

COUNTRY_LABELS = {
    "BR": "Brazilian Portuguese (português brasileiro)",
    "PT": "European Portuguese (português europeu)",
    "AO": "Angolan Portuguese (português angolano)",
}

COUNTRY_EXAMPLES = {
    "BR": (
        "Olha, organizar o dinheiro com dois filhos pequenos é um desafio real. "
        "O segredo é separar as despesas em grupos, assim você enxerga onde tá "
        "indo cada real. Tenta antecipar os gastos sazonais, como material "
        "escolar em fevereiro, e sempre deixar uns 10% de reserva pra imprevisto, "
        "que com criança é certo."
    ),
    "PT": (
        "Ola, recebi agora o subsidio de Natal e sao quase 900 euros, e a minha "
        "mulher ja me esta a dizer para trocar a televisao, mas eu acho que devia "
        "guardar uma parte, nao e? Pelo menos uns 30% para emergencias."
    ),
    "AO": (
        "Mano, aqui em Luanda a vida ta cara demais. O que eu faço é separar "
        "logo o dinheiro da renda e do transporte assim que recebo, e o resto "
        "divido entre comida e pequenas despesas. Sempre tento guardar um "
        "bocadinho no multicaixa."
    ),
}

SYSTEM_PROMPT = """\
You are a voice-message transcription generator for a Portuguese financial literacy dataset.
Your task: given a question and its reference answer about personal finance,
write a SHORT, COLLOQUIAL, ORAL-REGISTER transcription in {country_label}.

Rules:
- Write as if someone is SPEAKING into their phone (voice message / audio note)
- Use the dialect, slang, and expressions of {country_label}
- Keep it between 150 and 250 characters
- Do NOT copy the answer verbatim — rephrase it conversationally
- Include filler words, oral markers (e.g. "olha", "tipo", "entao", "mano", "pah")
- No punctuation perfection needed — mimic real oral speech
- The transcript must ANSWER the question, not repeat it
- Write ONLY the transcript text, nothing else

Example for {country_code}:
\"{example}\""""


def build_prompt(qa: QAPair) -> list[dict]:
    """Build the chat messages for GPT-4o-mini."""
    country_label = COUNTRY_LABELS[qa.country]
    example = COUNTRY_EXAMPLES[qa.country]
    system = SYSTEM_PROMPT.format(
        country_label=country_label,
        country_code=qa.country,
        example=example,
    )
    user = (
        f"Question: {qa.question}\n"
        f"Reference answer: {qa.answer}\n"
        f"Topic: {qa.topic} | Difficulty: {qa.difficulty}\n\n"
        f"Generate the oral audio transcript in {country_label}:"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# OpenAI API call (with retry)
# ---------------------------------------------------------------------------

def call_gpt(messages: list[dict], client, max_retries: int = 3) -> str:
    """Call GPT-4o-mini with retries."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=200,
                temperature=0.9,
            )
            text = response.choices[0].message.content.strip()
            # Remove surrounding quotes if present
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1]
            if text.startswith("'") and text.endswith("'"):
                text = text[1:-1]
            return text
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"    Retry {attempt+1}/{max_retries} after error: {e}. Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_file(
    jsonl_path: Path,
    client,
    sample: int | None = None,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Process a single JSONL file, adding audio_transcript where missing."""
    qa_pairs = load_jsonl(jsonl_path)
    country = qa_pairs[0].country if qa_pairs else "??"

    # Find pairs without audio
    missing = [(i, qa) for i, qa in enumerate(qa_pairs) if not qa.audio_transcript]
    existing = len(qa_pairs) - len(missing)
    print(f"\n  [{country}] {jsonl_path.name}: {len(qa_pairs)} total, "
          f"{existing} already have audio, {len(missing)} to generate")

    if sample is not None:
        missing = missing[:sample]
        print(f"    (sampling {len(missing)} for test)")

    generated = 0
    batch_size = 20  # save progress every 20

    for batch_start in range(0, len(missing), batch_size):
        batch = missing[batch_start:batch_start + batch_size]

        for idx, qa in batch:
            messages = build_prompt(qa)

            if dry_run:
                print(f"    [DRY] {qa.id}: would generate audio_transcript")
                continue

            transcript = call_gpt(messages, client)

            # Update QA pair
            qa_pairs[idx].audio_transcript = transcript
            if "audio" not in qa_pairs[idx].modalities:
                qa_pairs[idx].modalities.append("audio")

            generated += 1

            if generated % 10 == 0 or generated == len(missing):
                print(f"    [{country}] {generated}/{len(missing)} generated "
                      f"(last: {qa.id}, {len(transcript)} chars)")

            # Small delay to avoid rate limits
            time.sleep(0.1)

        # Save progress after each batch
        if not dry_run and batch:
            save_jsonl(qa_pairs, jsonl_path)
            print(f"    [{country}] Progress saved ({generated}/{len(missing)})")

    # Final save
    if not dry_run and generated > 0:
        save_jsonl(qa_pairs, jsonl_path)

    return generated, len(missing)


def main():
    parser = argparse.ArgumentParser(description="Expand audio transcripts to 100%")
    parser.add_argument("--sample", type=int, default=None,
                        help="Process only N items per file (for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't actually call the API")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable")
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    dataset_dir = _PROJECT_ROOT / "dataset"
    files = [
        dataset_dir / "BR" / "br_seed.jsonl",
        dataset_dir / "PT" / "pt_seed.jsonl",
        dataset_dir / "AO" / "ao_seed.jsonl",
    ]

    print("=" * 60)
    print("Expanding audio_transcript coverage to 100%")
    print("=" * 60)

    total_generated = 0
    total_missing = 0
    t0 = time.time()

    for f in files:
        if not f.exists():
            print(f"  WARNING: {f} not found, skipping")
            continue
        gen, miss = process_file(f, client, sample=args.sample, dry_run=args.dry_run)
        total_generated += gen
        total_missing += miss

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Done! Generated {total_generated}/{total_missing} audio transcripts in {elapsed:.1f}s")
    print(f"{'=' * 60}")

    # Verify final counts
    print("\nVerification:")
    for f in files:
        if f.exists():
            qa_pairs = load_jsonl(f)
            with_audio = sum(1 for q in qa_pairs if q.audio_transcript)
            print(f"  {f.name}: {with_audio}/{len(qa_pairs)} have audio_transcript")


if __name__ == "__main__":
    main()

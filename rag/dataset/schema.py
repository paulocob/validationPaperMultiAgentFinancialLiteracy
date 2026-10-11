"""
Schema de dados para o FinEduBench-Luso.

Define as estruturas de dados para pares de pergunta-resposta (QA)
no contexto de educacao financeira lusofona multimodal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class QAPair:
    """Par pergunta-resposta para educacao financeira lusofona."""

    id: str
    question: str                       # Pergunta em portugues
    answer: str                         # Resposta de referencia (gold)
    country: str                        # BR, PT, AO
    topic: str                          # budget, savings, debt, investment, credit, cultural
    difficulty: str                     # basic, intermediate, advanced
    modalities: list[str]               # modalidades relevantes: ["text"], ["text", "image"], etc.
    image_description: str | None = None   # descricao da imagem associada
    audio_transcript: str | None = None    # transcricao de audio associado
    cultural_notes: str | None = None      # notas sobre especificidade cultural
    sources: list[str] = field(default_factory=list)  # fontes de referencia

    def to_dict(self) -> dict[str, Any]:
        """Converte o QAPair para dicionario serializavel."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QAPair:
        """Cria um QAPair a partir de um dicionario."""
        return cls(**data)

    def to_json(self) -> str:
        """Serializa para string JSON."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> QAPair:
        """Desserializa a partir de string JSON."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class BenchmarkSplit:
    """Divisao do benchmark em conjuntos de treino, validacao e teste."""

    train: list[QAPair]
    val: list[QAPair]
    test: list[QAPair]

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        """Converte o split para dicionario."""
        return {
            "train": [qa.to_dict() for qa in self.train],
            "val": [qa.to_dict() for qa in self.val],
            "test": [qa.to_dict() for qa in self.test],
        }

    @classmethod
    def from_dict(cls, data: dict[str, list[dict[str, Any]]]) -> BenchmarkSplit:
        """Cria um BenchmarkSplit a partir de dicionario."""
        return cls(
            train=[QAPair.from_dict(d) for d in data["train"]],
            val=[QAPair.from_dict(d) for d in data["val"]],
            test=[QAPair.from_dict(d) for d in data["test"]],
        )


def save_jsonl(qa_pairs: list[QAPair], path: str | Path) -> None:
    """Salva lista de QAPair em formato JSONL (um JSON por linha)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qa in qa_pairs:
            f.write(qa.to_json() + "\n")


def load_jsonl(path: str | Path) -> list[QAPair]:
    """Carrega lista de QAPair a partir de arquivo JSONL."""
    path = Path(path)
    qa_pairs: list[QAPair] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qa_pairs.append(QAPair.from_json(line))
    return qa_pairs


def load_all_countries(base_dir: str | Path) -> list[QAPair]:
    """Carrega todos os QA pairs de todos os paises (BR, PT, AO)."""
    base_dir = Path(base_dir)
    all_pairs: list[QAPair] = []
    for country_code in ["BR", "PT", "AO"]:
        country_dir = base_dir / country_code
        if country_dir.exists():
            for jsonl_file in sorted(country_dir.glob("*.jsonl")):
                all_pairs.extend(load_jsonl(jsonl_file))
    return all_pairs


def make_split(
    qa_pairs: list[QAPair],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> BenchmarkSplit:
    """Divide os QA pairs em train/val/test mantendo proporcao por pais."""
    import random

    by_country: dict[str, list[QAPair]] = {}
    for qa in qa_pairs:
        by_country.setdefault(qa.country, []).append(qa)

    train_pairs: list[QAPair] = []
    val_pairs: list[QAPair] = []
    test_pairs: list[QAPair] = []

    for _country, pairs in by_country.items():
        shuffled = pairs.copy()
        random.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_pairs.extend(shuffled[:n_train])
        val_pairs.extend(shuffled[n_train : n_train + n_val])
        test_pairs.extend(shuffled[n_train + n_val :])

    return BenchmarkSplit(train=train_pairs, val=val_pairs, test=test_pairs)

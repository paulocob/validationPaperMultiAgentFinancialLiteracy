"""Expansoes do dataset FinEduBench-Luso (BR + PT + AO)."""
from dataset.schema import QAPair

from dataset.expansions.br_expansion import BR_EXPANSION
from dataset.expansions.pt_expansion import PT_EXPANSION
from dataset.expansions.ao_expansion import AO_EXPANSION
from dataset.expansions.br_expansion_2a import BR_EXPANSION_2A
from dataset.expansions.br_expansion_2b import BR_EXPANSION_2B
from dataset.expansions.br_expansion_2c import BR_EXPANSION_2C
from dataset.expansions.pt_expansion_2a import PT_EXPANSION_2A
from dataset.expansions.pt_expansion_2b import PT_EXPANSION_2B
from dataset.expansions.pt_expansion_2c import PT_EXPANSION_2C
from dataset.expansions.ao_expansion_2a import AO_EXPANSION_2A
from dataset.expansions.ao_expansion_2b import AO_EXPANSION_2B
from dataset.expansions.ao_expansion_2c import AO_EXPANSION_2C


def _to_qapairs(dicts: list[dict]) -> list[QAPair]:
    """Converte lista de dicts em lista de QAPair."""
    return [QAPair.from_dict(d) for d in dicts]


BR_EXPANSION_ALL = BR_EXPANSION + _to_qapairs(BR_EXPANSION_2A + BR_EXPANSION_2B + BR_EXPANSION_2C)
PT_EXPANSION_ALL = PT_EXPANSION + _to_qapairs(PT_EXPANSION_2A + PT_EXPANSION_2B + PT_EXPANSION_2C)
AO_EXPANSION_ALL = AO_EXPANSION + _to_qapairs(AO_EXPANSION_2A + AO_EXPANSION_2B + AO_EXPANSION_2C)

__all__ = [
    "BR_EXPANSION", "PT_EXPANSION", "AO_EXPANSION",
    "BR_EXPANSION_ALL", "PT_EXPANSION_ALL", "AO_EXPANSION_ALL",
]

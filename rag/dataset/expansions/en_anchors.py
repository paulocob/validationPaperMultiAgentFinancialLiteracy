"""Cross-lingual PT<->EN anchor pairs for CLIP fine-tuning alignment."""

from dataset.expansions.en_anchors_br import EN_ANCHORS_BR
from dataset.expansions.en_anchors_pt import EN_ANCHORS_PT
from dataset.expansions.en_anchors_ao import EN_ANCHORS_AO

EN_ANCHORS = EN_ANCHORS_BR + EN_ANCHORS_PT + EN_ANCHORS_AO

__all__ = ["EN_ANCHORS", "EN_ANCHORS_BR", "EN_ANCHORS_PT", "EN_ANCHORS_AO"]

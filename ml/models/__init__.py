"""Model architectures for depth learning."""

from .acoustic import ConvEncoder, TransformerEncoder
from .lstm import AcousticLSTM
from .ast import AcousticAST
from .jepa import CrossModalJEPA
from .lewm import LeWorldModel
from .lewm_plus import LeWMPlus  # LeWM++: JEPA + SigReg
from .mae import AcousticMAE
from .fusion import MaskedAttentionFusion
from .transformer_translator import AcousticToImageTransformer
from .decoder import LatentDecoder
from .clip import ProjectionHead, DualEncoderCLIP

__all__ = [
    # Acoustic Encoders
    "ConvEncoder",
    "TransformerEncoder",
    "AcousticLSTM",
    "AcousticAST",
    # Multi-modal Models
    "CrossModalJEPA",
    "LeWMPlus",  # LeWM++: Multi-modal JEPA + SigReg
    # Acoustic-only Models
    "LeWorldModel",
    "AcousticMAE",
    # Fusion & Translation
    "MaskedAttentionFusion",
    "AcousticToImageTransformer",
    "LatentDecoder",
    # CLIP-style
    "ProjectionHead",
    "DualEncoderCLIP",
]

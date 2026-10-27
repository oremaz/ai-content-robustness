"""Public text detector imports, retained for existing experiments and checkpoints."""

from .text_detectors_utils import _check_hf_config
from .diveye import DivEyeDetector
from .checkpoint_text import (
    ModernBERTDetector,
    TFIDFDetector,
)
from .likelihood_text import (
    BinocularsDetector,
    FastDetectGPTDetector,
)
from .rewrite_text import (
    ParaphraseRoundTripDetector,
    DisruptRecoverDetector,
    InversionDetector,
)
from .window_text import (
    WindowSmoothedTextDetector,
    MarkovCalibratedTextDetector,
)
from .ipad import (
    IPADDetector,
)

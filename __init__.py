from .nodes.audio_latent import NODE_CLASS_MAPPINGS as _ACM, NODE_DISPLAY_NAME_MAPPINGS as _ADM
from .nodes.calculators import NODE_CLASS_MAPPINGS as _CCM, NODE_DISPLAY_NAME_MAPPINGS as _CDM
from .nodes.audio_guide import NODE_CLASS_MAPPINGS as _AGM, NODE_DISPLAY_NAME_MAPPINGS as _AGD
from .nodes.sigmas import NODE_CLASS_MAPPINGS as _SCM, NODE_DISPLAY_NAME_MAPPINGS as _SDM
from .nodes.sigma_character import NODE_CLASS_MAPPINGS as _SCCM, NODE_DISPLAY_NAME_MAPPINGS as _SCDM
from .nodes.utils import NODE_CLASS_MAPPINGS as _UCM, NODE_DISPLAY_NAME_MAPPINGS as _UDM
from .nodes.av_looping_sampler import NODE_CLASS_MAPPINGS as _ALSCM, NODE_DISPLAY_NAME_MAPPINGS as _ALSDM
from .nodes.lora_train import NODE_CLASS_MAPPINGS as _LTCM, NODE_DISPLAY_NAME_MAPPINGS as _LTDM
from .nodes.character_dataset_prompt import NODE_CLASS_MAPPINGS as _CDPCM, NODE_DISPLAY_NAME_MAPPINGS as _CDPDM
from .nodes.speaker_ref import NODE_CLASS_MAPPINGS as _SRCM, NODE_DISPLAY_NAME_MAPPINGS as _SRDM
from .nodes.cross_attn_toggle import NODE_CLASS_MAPPINGS as _CATCM, NODE_DISPLAY_NAME_MAPPINGS as _CATDM
from .nodes.ref_audio_bank import NODE_CLASS_MAPPINGS as _RABCM, NODE_DISPLAY_NAME_MAPPINGS as _RABDM
from .nodes.video_cut_marker import NODE_CLASS_MAPPINGS as _VCMCM, NODE_DISPLAY_NAME_MAPPINGS as _VCMDM

NODE_CLASS_MAPPINGS        = {**_ACM, **_CCM, **_AGM, **_SCM, **_SCCM, **_UCM, **_ALSCM, **_LTCM, **_CDPCM, **_SRCM, **_CATCM, **_RABCM, **_VCMCM}
NODE_DISPLAY_NAME_MAPPINGS = {**_ADM, **_CDM, **_AGD, **_SDM, **_SCDM, **_UDM, **_ALSDM, **_LTDM, **_CDPDM, **_SRDM, **_CATDM, **_VCMDM, **_RABDM}

WEB_DIRECTORY = "./web/js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

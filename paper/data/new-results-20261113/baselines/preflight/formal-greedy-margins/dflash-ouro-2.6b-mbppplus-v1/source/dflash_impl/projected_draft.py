"""Narrow DFlash adapter; official transformer blocks and loss stay unchanged.

The target embedding/head remain external and frozen. Trainable projections
connect target-width features and embeddings to a narrower drafter, then project
its output back to the target head width. This is our adaptation, not an upstream
checkpoint architecture or an already-validated speed improvement.
"""
from copy import deepcopy

from torch import nn


def projected_config(target_config, hidden_size=2048, intermediate_size=5632, heads=16):
    cfg = deepcopy(target_config)
    target_width = cfg.hidden_size
    assert hidden_size > 0 and hidden_size % heads == 0
    assert cfg.num_hidden_layers == 5 and cfg.block_size == 16
    cfg.hidden_size = hidden_size
    cfg.intermediate_size = intermediate_size
    cfg.num_attention_heads = cfg.num_key_value_heads = heads
    cfg.head_dim = hidden_size // heads
    cfg.dflash_config = dict(cfg.dflash_config, target_hidden_size=target_width,
        projection_adapter='target-width-projections-v1')
    cfg.architectures = ['ProjectedDFlashDraftModel']
    return cfg


class TargetWidthProjections:
    def __init__(self, config):
        super().__init__(config)
        assert config.dflash_config['projection_adapter'] == 'target-width-projections-v1'
        self.target_hidden_size = config.dflash_config['target_hidden_size']
        self.fc = nn.Linear(len(self.target_layer_ids)*self.target_hidden_size,
                            config.hidden_size, bias=False)
        self.input_projection = nn.Linear(self.target_hidden_size, config.hidden_size, bias=False)
        self.output_projection = nn.Linear(config.hidden_size, self.target_hidden_size, bias=False)
        # Hugging Face initializes only new/uninitialized modules here. This also
        # respects its no-init/meta-device context when loading saved checkpoints.
        self.post_init()

    def forward(self, position_ids, attention_mask=None, noise_embedding=None, target_hidden=None, **kwargs):
        hidden = super().forward(position_ids=position_ids, attention_mask=attention_mask,
            noise_embedding=self.input_projection(noise_embedding), target_hidden=target_hidden, **kwargs)
        return self.output_projection(hidden)


def training_draft_class():
    from specforge.modeling.draft.dflash import DFlashDraftModel

    class ProjectedDFlashDraftModel(TargetWidthProjections, DFlashDraftModel):
        pass

    return ProjectedDFlashDraftModel


def inference_draft_class():
    from dflash.model import DFlashDraftModel

    class ProjectedDFlashDraftModel(TargetWidthProjections, DFlashDraftModel):
        pass

    return ProjectedDFlashDraftModel

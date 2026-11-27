from abc import ABC, abstractmethod

import torch

from elasticai.creator.nn.integer.quant_utils.QParams import QParams
from elasticai.creator.vhdl.design.design import Design


class DesignCreatorModule(ABC, torch.nn.Module):
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Earlier integer layers stored derived bias parameters as a child module.
        if "bias_QParams" not in self._modules:
            legacy_prefix = prefix + "bias_QParams."
            for key in tuple(state_dict):
                if key.startswith(legacy_prefix):
                    del state_dict[key]
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @abstractmethod
    def create_design(self, name: str) -> Design: ...

    @abstractmethod
    def forward(
        self, inputs: torch.Tensor, given_inputs_QParams: QParams | None = None
    ) -> torch.Tensor: ...

    @abstractmethod
    def int_forward(self, q_inputs: torch.Tensor) -> torch.Tensor: ...

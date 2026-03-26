from pathlib import Path

import elasticai.experiment_framework.remote_control as eaixp_rc
import pytest
from elasticai.experiment_framework.synthesis import CachedVivadoSynthesis

from elasticai.creator.testing import HWTester
from elasticai.creator_plugins.xnor_popcount_mac.tests._common import (
    CNNBuilder,
    build_design,
    prepare_ir2vhdl_for_hw,
)


def synthesize(src_dir: Path) -> Path:
    _synth = CachedVivadoSynthesis()
    return _synth.synthesize(src_dir) / "results/impl/env5_top_reconfig.bin"


@pytest.mark.hardware
def test_run_minimal_binary_cnn_defined_in_low_level_ir_on_hardware(tmp_path):
    data_depth = 4

    weight = "10"
    kernel_size = len(weight)
    expected_output_words = b"\x01\x01\x00"

    # Use CNNBuilder instead of build_network
    builder = CNNBuilder(data_out_depth=data_depth - kernel_size + 1)
    builder.add_conv(weight)
    registry = builder.build()

    build_dir = tmp_path / "vhdl"
    if not build_dir.exists():
        build_dir.mkdir()

    device = eaixp_rc.probe_for_devices()[0]
    ctx = HWTester(
        synth_fn=synthesize,
        device=eaixp_rc.remote_control.connect_remote_control(device),
    )
    hw_id = build_design(
        registry["network"], registry, build_dir, ir2vhdl=prepare_ir2vhdl_for_hw()
    )

    with ctx.prepare_hw_function(build_dir, id=hw_id) as run_inference:
        predictions = run_inference(b"\x01\x01\x00\x01", 3)
    assert predictions == expected_output_words

import os
import shutil
from pathlib import Path
from time import sleep

import elasticai.experiment_framework.remote_control as eaixp_rc
import pytest
from elasticai.experiment_framework.synthesis import run_synthesis as _run_synthesis
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from elasticai.creator import ir
from elasticai.creator.ir2vhdl import Ir2Vhdl, IrFactory, PluginLoader
from elasticai.creator_plugins.skeleton.hw_function_id import (
    HwFunctionIdUpdater as SkeletonHwFunctionIdUpdater,
)


@pytest.mark.hardware
@given(data=st.data())
@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=10000,
)
@pytest.mark.parametrize(["word_size", "num_features"], [(5, 4)])
def test_skeleton_identity_system(tmp_path: Path, word_size: int, num_features, data):
    # Each inner list is one inference sample.
    sample = data.draw(
        st.lists(st.integers(min_value=0), min_size=num_features, max_size=num_features)
    )

    sample = b"".join([_to_bytes(value, word_size) for value in sample])
    output_num_bytes = len(sample)
    build_dir = tmp_path / f"identity_skeleton_build_w{word_size}"
    skeleton_id = _create_identity_vhdl(build_dir, word_size, num_features)

    binfile = _run_synthesis_cached(build_dir / "srcs", skeleton_id)

    flash_sector = 0
    predictions = []
    discovered_devices = eaixp_rc.devices.probe_for_devices()
    device = discovered_devices[0]
    with device.connect() as serial_con:
        rc = eaixp_rc.RemoteControl(serial_con)
        # Upload requires FPGA power off.
        rc.fpga_power_on()
        sleep(0.1)
        hw_function_id = rc.read_skeleton_id()
        if hw_function_id != skeleton_id.hex(" "):
            rc.fpga_power_off()
            sleep(0.1)
            rc.upload_bitstream(flash_sector, str(binfile))
            # Interacting with the hardware implementation requires FPGA power on.
            rc.fpga_power_on()
            sleep(0.1)
            id_readback = rc.read_skeleton_id()

            assert id_readback == skeleton_id.hex(" ")

        predictions.append(rc.predict(sample, output_num_bytes))

    numeric_sample = bytes_to_words_padded(sample, word_size, num_features)
    prediction = bytes_to_words_padded(predictions[0], word_size, num_features)
    assert prediction == numeric_sample
    assert predictions[0] == sample


def identity_network(graph, _registry):
    width = int(graph.attributes["data_width"])
    name = graph.name
    lines = (
        "library ieee;",
        "use ieee.std_logic_1164.all;",
        "",
        f"entity {name} is",
        "  port (",
        "    CLK : in std_logic;",
        f"    D_IN : in std_logic_vector({width - 1} downto 0);",
        f"    D_OUT : out std_logic_vector({width - 1} downto 0);",
        "    SRC_VALID : in std_logic;",
        "    RST : in std_logic;",
        "    VALID : out std_logic;",
        "    READY : out std_logic;",
        "    DST_READY : in std_logic;",
        "    EN : in std_logic",
        "  );",
        "end entity;",
        "",
        f"architecture rtl of {name} is",
        "begin",
        "  D_OUT <= D_IN;",
        "  VALID <= SRC_VALID and DST_READY and EN;",
        "  READY <= DST_READY;",
        "end architecture;",
    )
    return ((name, lines),)


def _prepare_ir2vhdl():
    translator = Ir2Vhdl()
    plugins = PluginLoader(translator)
    plugins.load_from_package("skeleton")
    plugins.load_from_package("middleware")
    translator.register("identity_network", identity_network)
    return translator


def _create_identity_vhdl(output_dir: Path, word_size: int, data_depth: int) -> bytes:
    srcs_dir = output_dir / "srcs"
    srcs_dir.mkdir(parents=True, exist_ok=True)
    translator = _prepare_ir2vhdl()
    factory = IrFactory()

    network = factory.graph(
        ir.attribute(data_width=word_size),
        type="identity_network",
        name="network",
    )
    skeleton = factory.graph(
        ir.attribute(
            generic_map={
                "DATA_IN_WIDTH": str(word_size),
                "DATA_IN_DEPTH": str(data_depth),
                "DATA_OUT_WIDTH": str(word_size),
                "DATA_OUT_DEPTH": str(data_depth),
            }
        ),
        type="skeleton",
        name="skeleton",
    )

    code = translator(
        skeleton,
        ir.Registry((("network", network),)),
    )

    for filename, lines in code:
        (srcs_dir / filename).write_text("\n".join(lines) + "\n")

    hwid_updater = SkeletonHwFunctionIdUpdater(srcs_dir)
    hwid_updater.compute_id()
    hwid_updater.write_id()
    return hwid_updater.id


def _cached_hw_function_id_matches(cache_entry: Path, hw_function_id: bytes) -> bool:
    cached_id_file = cache_entry / "hw_function_id.txt"
    if not cached_id_file.exists():
        return False
    return cached_id_file.read_text().strip() == hw_function_id.hex()


def _run_synthesis_cached(src_dir: Path, hw_function_id: bytes) -> Path:
    cache_root = Path(
        os.getenv(
            "ELASTICAI_HW_SYNTH_CACHE_DIR",
            str(Path.home() / ".cache" / "elasticai" / "synthesis" / "skeleton"),
        )
    )
    cache_entry = cache_root / hw_function_id.hex()
    cached_bin = cache_entry / "design.bin"
    if cached_bin.exists() and _cached_hw_function_id_matches(
        cache_entry, hw_function_id
    ):
        return cached_bin

    cache_entry.mkdir(parents=True, exist_ok=True)
    synthesized_bin = _run_synthesis(src_dir)
    shutil.copy2(synthesized_bin, cached_bin)
    (cache_entry / "hw_function_id.txt").write_text(hw_function_id.hex() + "\n")
    return cached_bin


def _transport_word_size(word_size: int) -> int:
    return max(8, ((word_size + 7) // 8) * 8)


def _normalize_word(value: int, word_size: int) -> int:
    mask = (1 << word_size) - 1
    encoded = int(value) & mask
    return encoded


def _to_bytes(value: int, word_size: int) -> bytes:
    encoded = _normalize_word(value, word_size)
    byte_count = max(1, (word_size + 7) // 8)
    return encoded.to_bytes(length=byte_count, byteorder="little", signed=False)


def bytes_to_words_padded(data: bytes, word_size: int, n_words: int) -> list[int]:
    transport_bits = ((word_size + 7) // 8) * 8
    packed = int.from_bytes(data, byteorder="little", signed=False)
    mask = (1 << word_size) - 1
    return [(packed >> (i * transport_bits)) & mask for i in range(n_words)]

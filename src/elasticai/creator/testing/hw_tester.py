import contextlib
import logging
from abc import abstractmethod
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from time import sleep
from typing import Protocol


class _SynthesisFn(Protocol):
    def __call__(self, src_dir: Path) -> Path: ...


class AIAccelerator(Protocol):
    @abstractmethod
    def __call__(self, input_data: bytes, result_size: int) -> bytes: ...


class _RemoteControl(Protocol):
    def predict(self, data: bytes, result_size: int) -> bytes: ...
    def fpga_power_on(self) -> None: ...
    def fpga_power_off(self) -> None: ...
    def read_skeleton_id(self) -> bytes | str: ...
    def upload_bitstream(self, flash_sector: int, path_to_bitstream: str) -> None: ...


class _AIAcceleratorImpl:
    def __init__(self, rc: _RemoteControl):
        self._rc = rc

    def __call__(self, input_data: bytes, result_size: int) -> bytes:
        """Execute inference on hardware."""
        return self._rc.predict(input_data, result_size)


class HWTester:
    """Prepare HW Functions for testing.

    Usage example:

    ```python
    import elasticai.experiment_framework as eaixp
    import elasticai.creator.testing as crt
    from pathlib import Path
    import pytest

    @pytest.fixture
    def hw_tester():
        synthesis = eaixp.synthesis.CachedVivadoSynthesis()
        def synthesize(src_dir: Path) -> Path:
            return synthesis.synthesize(src_dir) / "results/impl/env5_top_reconfig.bin"
        device = eaixp.remote_control.probe_for_devices()[0]

        ctx = HWTesterContext(synthesize,
            eaixp.remote_control.connect_remote_control(device))
        return ctx

    def test_my_fn(hw_tester, tmp_dir):
        build_dir = tmp_dir / "build"
        expected_id = generate_srcs(build_dir)
        with hw_tester.prepare_hw_function(build_dir, expected_id) as my_fn:
            result = my_fn(b"\xde\xad\xbe\xef", 1)
        assert result = b"\x12"s
    ```
    """

    def __init__(
        self, synth_fn: _SynthesisFn, device: AbstractContextManager[_RemoteControl]
    ):
        self._synth = synth_fn
        self._device = device
        self._ctx_manager = contextlib.ExitStack()
        self._logger = logging.getLogger("eai.HWTester")
        self._binfile: Path | None = None
        self._id: bytes | None = None
        self._rc: _RemoteControl | None = None

    def _get_rc(self) -> _RemoteControl:
        if self._rc is None:
            raise RuntimeError("remote control not connected")
        return self._rc

    @contextmanager
    def prepare_hw_function(
        self, src_dir: Path, id: bytes | None = None
    ) -> Generator[AIAccelerator]:
        self._logger.debug("preparing hw function")
        self._logger.debug(f"synthesizing project in {src_dir.absolute()}")
        self._binfile = self._synth(src_dir)
        self._logger.debug(f"synthesized project to binfile {self._binfile.absolute()}")
        self._id = id
        with self._device as rc:
            self._rc = rc
            self._upload()
            self._logger.debug("hw function prepared")
            yield _AIAcceleratorImpl(self._rc)
            self._logger.debug("closing HW Tester")
            self._power_off()
        self._rc = None

    def _power_on(self) -> None:
        self._get_rc().fpga_power_on()

    def _power_off(self) -> None:
        self._get_rc().fpga_power_off()

    def _get_binfile(self) -> Path:
        if self._binfile is None:
            raise RuntimeError("no binfile specified")
        return self._binfile

    def _do_upload(self) -> None:
        self._get_rc().upload_bitstream(
            flash_sector=0, path_to_bitstream=str(self._get_binfile())
        )

    def _upload(self) -> None:
        """Upload the hardware function to the device."""
        self._power_on()
        sleep(0.1)
        self._retrieve_loaded_id()
        if self._id is None or not self._is_loaded():
            self._logger.debug("uploading binfile to device")
            self._power_off()
            sleep(0.1)
            self._do_upload()
            sleep(0.1)
            self._power_on()
            sleep(0.2)
            self._ensure_is_loaded()
        else:
            self._logger.debug("skipped upload: binfile already present")

    def _ensure_is_loaded(self) -> None:
        self._retrieve_loaded_id()
        if not self._is_loaded():
            assert self._id is not None
            assert self._binfile is not None
            raise RuntimeError(
                f"failed to load HW function to FPGA, expected function id {self._id.hex(' ')}, but found {self._current_hw_id.hex(' ')}\nloaded file was {self._binfile.absolute()}"
            )

    def _retrieve_loaded_id(self) -> None:
        _id = self._get_rc().read_skeleton_id()
        if isinstance(_id, str):
            self._current_hw_id = bytes.fromhex(_id)

    def _is_loaded(self) -> bool:
        """Check if the hardware function is loaded."""
        assert self._id is not None
        return self._current_hw_id == self._id

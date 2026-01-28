import logging
from collections.abc import Callable, Iterable, Iterator
from hashlib import blake2b, file_digest
from pathlib import Path


class HwFunctionIdUpdater:
    """Hash all files in `build_dir` except for `target_file` and uses `replace_id_fn` to write the result to `target_file`.

    The id will stay the same, if none of the files, except for the target file changed.

    You can use the optional `file_filter` to exclude additional files from the id computation.
    Example:

    ```python
    generate_my_source_files(build_dir)
    def replace_id(code: Iterable[str], id: bytes) -> Iterator[str]:
        for line in code:
            if line == "my_id_placeholder":
                yield f"my_id = {id.hex()};"
            else:
                yield line

    def filter_non_vhd(file):
        return str(file).endswith(".vhd")

    id_updater = HwFunctionIdUpdate(build_dir, "id_package.vhd", replace_id, filter_non_vhd)
    id_update.compute_id()
    id_updater.write_id()
    print(id_updater.id)
    ```

    """

    def __init__(
        self,
        build_dir: Path,
        target_file: str | Path,
        replace_id_fn: Callable[[Iterable[str], bytes], Iterator[str]],
        file_filter: Callable[[Path], bool] = lambda f: True,
    ):
        self._target_file = Path(target_file)
        self._build_dir = build_dir
        self._id = bytes()
        self._file_filter = file_filter
        self._replace_id_fn = replace_id_fn

    def _files_from_build_dir(self) -> Iterator[Path]:
        for f in self._build_dir.glob("**/*"):
            if f.is_file() and self._file_filter(f):
                yield f

    def _is_not_target_file(self, file: Path) -> bool:
        return not file.samefile(self._target_file)

    def _collect_files_to_hash(self) -> Iterator[Path]:
        return filter(self._is_not_target_file, self._files_from_build_dir())

    def compute_id(self) -> None:
        logger = logging.getLogger(__name__)
        hash = _HwFunctionIdHash()

        for vhd_file in self._collect_files_to_hash():
            logger.debug(f"hashing {vhd_file.as_uri}")
            hash.update(vhd_file)
        digest = hash.digest()
        logger.debug(f"raw_digest is {digest.hex()}")

        self._id = digest

    def write_id(self) -> None:
        with open(self._target_file, "r") as f:
            code: Iterable[str] = f.readlines()
        code = self._replace_id_fn(code, self.id)
        with open(self._target_file, "w") as f:
            for line in code:
                f.write(line)
                f.write("\n")

    @property
    def id(self) -> bytes:
        return self._id


class _HwFunctionIdHash:
    SIZE: int = 16

    def __init__(self):
        self._digests = []

    def _hash(self):
        return blake2b(digest_size=self.SIZE)

    def update(self, lines: Iterable[str] | Path) -> None:
        h = self._hash()

        if isinstance(lines, Path):
            with open(lines, "rb") as f:
                self._digests.append(file_digest(f, self._hash).digest())
        else:
            for line in lines:
                h.update(line.encode())
            self._digests.append(h.digest())

    def digest(self) -> bytes:
        h = self._hash()
        for d in sorted(self._digests):
            h.update(d)
        return h.digest()

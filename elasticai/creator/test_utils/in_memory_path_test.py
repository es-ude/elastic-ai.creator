import pytest

from .in_memory_path import InMemoryPath


@pytest.fixture
def path() -> InMemoryPath:
    return InMemoryPath()


def test_init_using_single_string_or_mutiple_segments_is_equal() -> None:
    p1 = InMemoryPath("a/b/c/d")
    p2 = InMemoryPath("a", "b/c", "d")
    assert str(p1) == str(p2)


def test_string_representation_is_correct() -> None:
    path = InMemoryPath("hello", "world")
    assert str(path) == "hello/world"


@pytest.mark.parametrize("path", [InMemoryPath(""), InMemoryPath("a")])
def test_too_short_path_raises_error_when_accessing_parent(path: InMemoryPath) -> None:
    with pytest.raises(ValueError):
        path.parent


@pytest.mark.parametrize(
    "path, expected", [(InMemoryPath("a/b"), "a"), (InMemoryPath("a/b/c"), "a/b")]
)
def test_parent_returns_always_second_last_segment(
    path: InMemoryPath, expected: str
) -> None:
    assert str(path.parent) == expected


def test_empty_path_raises_error_when_accessing_name(path: InMemoryPath) -> None:
    with pytest.raises(ValueError):
        path.name


@pytest.mark.parametrize(
    "path, expected", [(InMemoryPath("a"), "a"), (InMemoryPath("a/b"), "b")]
)
def test_name_returns_always_last_segment(path: InMemoryPath, expected: str) -> None:
    assert path.name == expected


def test_joinpath_and_truediv_concatenates_paths() -> None:
    p1 = InMemoryPath("a/b")
    p2 = InMemoryPath("c/d")
    assert p1.joinpath(str(p2))
    p1.joinpath(str(p2))


def test_no_files_for_freshly_initialized_path(path: InMemoryPath) -> None:
    assert path.files == dict()

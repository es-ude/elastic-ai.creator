import os

from hypothesis import settings

pytest_plugins = ["elasticai.creator.testing"]

settings.register_profile("fast", max_examples=25, deadline=None)
settings.register_profile("ci", max_examples=100)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "ci"))


def pytest_sessionstart(session):
    from shutil import rmtree

    from elasticai.creator.file_generation import find_project_root

    folders = ["build_test", "build_sim"]

    for folder in folders:
        path2temp = find_project_root() / folder
        if path2temp.exists():
            rmtree(path2temp, ignore_errors=True)
        path2temp.mkdir(parents=True, exist_ok=True)


def pytest_sessionfinish(session, exitstatus):
    from pathlib import Path

    from elasticai.creator.file_generation import find_project_root

    def remove_empty_dirs(path: Path) -> None:
        if not path.exists():
            return
        for dirpath in path.iterdir():
            try:
                dirpath.rmdir()
            except:
                pass

    path2check = find_project_root() / "build_test"
    remove_empty_dirs(path2check)

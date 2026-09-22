pytest_plugins = ["elasticai.creator.testing"]


def pytest_sessionstart(session):
    from elasticai.creator.file_generation import find_project_root
    from shutil import rmtree

    path2temp = find_project_root() / "build_test"
    if path2temp.exists():
        rmtree(path2temp, ignore_errors=True)
    path2temp.mkdir(parents=True, exist_ok=True)


def pytest_sessionfinish(session, exitstatus):
    from elasticai.creator.file_generation import find_project_root
    from pathlib import Path

    path2check = find_project_root() / "build_test"
    def remove_empty_dirs(path: Path) -> None:
        if not path.exists():
            return
        for dirpath in path2check.iterdir():
            try:
                dirpath.rmdir()
            except:
                pass
    remove_empty_dirs(path2check)

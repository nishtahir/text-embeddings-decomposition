import os

__dirname__ = os.path.dirname(os.path.realpath(__file__))
__PROJECT_ROOT__ = os.path.join(__dirname__, "../")


def ensure_directories(paths: list[str]):
    for path in paths:
        if not os.path.exists(resolve_path(path)):
            os.makedirs(resolve_path(path))


def resolve_path(paths: list[str] | str) -> str:
    if isinstance(paths, str):
        return os.path.join(__PROJECT_ROOT__, paths)
    else:
        return os.path.join(__PROJECT_ROOT__, *paths)

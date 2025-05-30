import re
from functools import lru_cache
from importlib.resources import files


@lru_cache(maxsize=None)
def read_abnf(name: str) -> str:
    if not re.compile(r"^\w+$").fullmatch(name):
        raise ValueError(f"Invalid ABNF name: {name!r}")
    path = files(__name__).joinpath(f"{name}.abnf")
    with path.open(encoding="utf-8") as f:
        return f.read()

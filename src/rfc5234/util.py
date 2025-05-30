from collections.abc import Generator


def consume[E, R](gen: Generator[E, None, R]) -> tuple[R, list[E]]:
    rs: list[E] = list()
    try:
        while True:
            value = next(gen)
            rs.append(value)
    except StopIteration as e:
        return e.value, rs

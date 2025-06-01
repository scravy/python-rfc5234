from collections import ChainMap

from rfc5234.context import Context


def test_fork_join():
    ctx = Context(
        ChainMap({"foo": None}),
        ChainMap({"qux": 0}),
    )
    local_ctx = ctx.fork()
    local_ctx.push("foo", 8)
    assert local_ctx.top("foo") == 8
    assert ctx.top("foo") is None
    ctx.join(local_ctx)
    assert ctx.top("foo") == 8

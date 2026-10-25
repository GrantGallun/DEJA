"""The demo owns a connection; each operation owns its mutable model settings."""
from types import SimpleNamespace
import threading

import pytest

from smartreplay.demo import server, search


class IdleNative:
    def __init__(self, *args):
        self.closed = threading.Event()

    def status(self):
        return {"available": False, "enabled": False, "recording": False}

    def close(self):
        self.closed.set()


def test_default_factory_shares_transport_but_not_mutable_model(tmp_path, monkeypatch):
    calls = []
    injected = []
    transport = SimpleNamespace(close=lambda: calls.append("closed"))

    def factory(backend, directory, *, transport=None):
        calls.append(backend)
        injected.append(transport)
        return SimpleNamespace(transport=transport or connection, timeout=60, _owns_transport=True)

    connection = transport
    monkeypatch.setattr(server, "create_model", factory)
    session = server.DemoSession(tmp_path, native_factory=IdleNative)
    try:
        first, second = session.model_factory(), session.model_factory()
        assert first is not second
        assert first.transport is second.transport
        assert injected == [None, connection]
        assert not first._owns_transport and not second._owns_transport
        first.timeout = 15
        assert second.timeout == 60
        session.set_backend("claude")
        assert calls.count("closed") == 1
        session.set_backend("codex")
        session.model_factory()
    finally:
        session.close()
    assert calls.count("closed") == 2
    with pytest.raises(RuntimeError, match="closed"):
        session.model_factory()


def test_model_close_cannot_delay_native_shutdown(tmp_path, monkeypatch):
    order = []
    class Native(IdleNative):
        def close(self):
            order.append("native")
            super().close()
    transport = SimpleNamespace(close=lambda: order.append("model"))
    monkeypatch.setattr(server, "create_model", lambda *a, **kw: SimpleNamespace(transport=transport))
    session = server.DemoSession(tmp_path, native_factory=Native)
    session.model_factory()
    session.close()
    session.close()
    assert order == ["native", "model"]


def test_old_index_factory_cannot_open_replacement_connection(tmp_path):
    class Queries:
        def __init__(self, root, factory, snapshot):
            self.factory = factory
        def close(self):
            pass
    class Native(IdleNative):
        def snapshot(self, *args):
            raise AssertionError("No media needed")
    session = server.DemoSession(tmp_path, model_factory=lambda: object(),
                                 native_factory=Native, session_search_factory=Queries)
    try:
        old = session._queries()
        assert old.factory() is not None
        session.set_backend("claude")
        with pytest.raises(RuntimeError, match="changed"):
            old.factory()
        current = session._queries()
        assert current.factory() is not None
    finally:
        session.close()
    with pytest.raises(RuntimeError, match="closed"):
        current.factory()


def test_codex_model_passes_ordered_images_schema_and_local_timeout(tmp_path):
    calls = []
    class Transport:
        def request(self, prompt, image_paths, schema, **kwargs):
            calls.append((prompt, [p.read_bytes() for p in image_paths], schema, kwargs))
            return {"matches": [], "note": "No event"}
    model = search.CodexVisionModel(tmp_path, binary="codex", transport=Transport())
    model.timeout = 15
    response = model._call([{"type": "text", "text": "Frame at 1.250 seconds:"},
                           {"type": "image", "source": {"data": "aGVsbG8="}}])
    assert response["matches"] == []
    prompt, images, schema, kwargs = calls[0]
    assert "1.250" in prompt and "image 1" in prompt
    assert images == [b"hello"] and schema == search.RESULT_SCHEMA
    assert kwargs == {"model": "gpt-6-luna", "timeout": 15}

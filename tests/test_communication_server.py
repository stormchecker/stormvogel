"""Tests for stormvogel.communication_server."""

import json
import socket
import string
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

import stormvogel.communication_server as cs


# ── Helpers ───────────────────────────────────────────────────────────


def _free_port() -> int:
    """OS-assigned free TCP port number, released immediately."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _post(port: int, event_id: str, data) -> None:
    """Fire-and-forget HTTP POST, mirroring what the JS fetch() does.

    Sends the request and closes the socket without waiting for a response,
    because the server handler doesn't send one.
    """
    payload = json.dumps({"id": event_id, "data": data}).encode()
    header = (
        f"POST / HTTP/1.1\r\n"
        f"Host: 127.0.0.1\r\n"
        f"Content-Length: {len(payload)}\r\n\r\n"
    ).encode()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(2)
        s.connect(("127.0.0.1", port))
        s.sendall(header + payload)


def _wait_for_port(port: int, timeout: float = 2.0) -> None:
    """Block until the port is accepting TCP connections."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.socket() as s:
                s.settimeout(0.1)
                s.connect(("127.0.0.1", port))
            return
        except (ConnectionRefusedError, OSError):
            time.sleep(0.01)
    raise TimeoutError(f"Server did not start on port {port} within {timeout}s")


def _wait_for(condition, timeout: float = 2.0) -> None:
    """Spin until condition() is truthy or timeout elapses."""
    deadline = time.time() + timeout
    while not condition() and time.time() < deadline:
        time.sleep(0.01)


@pytest.fixture
def comm_server():
    """CommunicationServer on a free port with IPython mocked out.

    Saves and restores all module-level globals so tests don't leak state.
    """
    port = _free_port()
    saved_events = dict(cs.events)
    saved_server = cs.server
    saved_running = cs.server_running

    with (
        patch("IPython.display.display"),
        patch("IPython.display.HTML"),
        patch("IPython.display.Javascript"),
    ):
        srv = cs.CommunicationServer(server_port=port)

    _wait_for_port(port)

    cs.server = srv
    cs.server_running = True
    cs.events.clear()

    yield srv

    srv.web_server.shutdown()
    cs.events.clear()
    cs.events.update(saved_events)
    cs.server = saved_server
    cs.server_running = saved_running


# ── random_word ───────────────────────────────────────────────────────


def test_random_word_length():
    for k in (1, 10, 20):
        assert len(cs.random_word(k)) == k


def test_random_word_ascii_letters_only():
    word = cs.random_word(100)
    assert all(c in string.ascii_letters for c in word)


def test_random_word_unique():
    # k=20 gives 52^20 ≈ 10^34 possibilities; collision probability is negligible
    assert cs.random_word(20) != cs.random_word(20)


# ── is_port_free / find_free_port ────────────────────────────────────


def test_is_port_free_for_unbound_port():
    port = _free_port()
    assert cs.is_port_free(port)


def test_is_port_free_for_occupied_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)  # is_port_free uses connect_ex, so the port must be listening
        port = s.getsockname()[1]
        assert not cs.is_port_free(port)


def test_find_free_port_returns_value_in_range():
    port = cs.find_free_port()
    if port != -1:
        assert port in cs.port_range


def test_find_free_port_returns_minus_one_when_all_occupied():
    sockets = []
    try:
        for p in cs.port_range:
            try:
                s = socket.socket()
                s.bind(("127.0.0.1", p))
                s.listen(1)  # is_port_free uses connect_ex, so must be listening
                sockets.append(s)
            except OSError:
                pass  # Already occupied externally — still counts as unavailable

        if len(sockets) == len(cs.port_range):
            assert cs.find_free_port() == -1
        else:
            pytest.skip("Could not occupy all ports in the configured range")
    finally:
        for s in sockets:
            s.close()


# ── HTTP dispatch (direct events dict access) ─────────────────────────


def test_post_triggers_registered_callback(comm_server):
    """A raw HTTP POST with a known id invokes the registered callback."""
    received = []
    cs.events["dispatch-test-1"] = lambda data: received.append(json.loads(data))

    _post(comm_server.server_port, "dispatch-test-1", {"value": 42})

    _wait_for(lambda: received)
    assert received == [{"value": 42}]


def test_post_passes_data_through_json_dumps(comm_server):
    """The server wraps the POST body's data field in json.dumps before calling the callback."""
    received = []
    cs.events["dispatch-test-2"] = lambda data: received.append(data)

    _post(comm_server.server_port, "dispatch-test-2", "hello")

    _wait_for(lambda: received)
    # do_POST does: data = json.dumps(body["data"]), so "hello" → '"hello"'
    assert received == ['"hello"']


def test_multiple_posts_trigger_separate_callbacks(comm_server):
    received_a, received_b = [], []
    cs.events["multi-a"] = lambda data: received_a.append(data)
    cs.events["multi-b"] = lambda data: received_b.append(data)

    _post(comm_server.server_port, "multi-a", 1)
    _post(comm_server.server_port, "multi-b", 2)

    _wait_for(lambda: received_a and received_b)
    assert received_a == ["1"]
    assert received_b == ["2"]


# ── add_event ─────────────────────────────────────────────────────────


def test_add_event_registers_callback_in_events(comm_server):
    def fn(data):
        pass

    with patch("IPython.display.display"), patch("IPython.display.Javascript"):
        event_id = comm_server.add_event("FUNCTION(1+1);", fn)

    assert event_id in cs.events
    assert cs.events[event_id] is fn
    cs.events.pop(event_id)


def test_add_event_rewrites_function_call_to_post_url(comm_server):
    """FUNCTION( is replaced with return_id_result(...) pointing to the server."""
    captured_js = []

    with (
        patch("IPython.display.display"),
        patch(
            "IPython.display.Javascript",
            side_effect=lambda js: captured_js.append(js) or MagicMock(),
        ),
    ):
        event_id = comm_server.add_event("FUNCTION(x);", lambda d: None)

    assert captured_js, "Javascript() was never called"
    js = captured_js[0]
    assert f"127.0.0.1:{comm_server.server_port}" in js
    assert f"'{event_id}'" in js
    assert "FUNCTION(" not in js
    cs.events.pop(event_id)


def test_add_event_raises_when_global_server_is_none(comm_server):
    original = cs.server
    cs.server = None
    try:
        with pytest.raises(TimeoutError, match="no server"):
            comm_server.add_event("FUNCTION(1);", lambda d: None)
    finally:
        cs.server = original


# ── remove_event ──────────────────────────────────────────────────────


def test_remove_event_returns_and_removes_callback(comm_server):
    def fn(data):
        pass

    cs.events["remove-test"] = fn

    returned = comm_server.remove_event("remove-test")

    assert returned is fn
    assert "remove-test" not in cs.events


def test_remove_event_raises_for_unknown_id(comm_server):
    with pytest.raises(KeyError):
        comm_server.remove_event("does-not-exist")


# ── result ────────────────────────────────────────────────────────────


def test_result_returns_data_from_simulated_js(comm_server):
    """result() blocks until a POST arrives; a background thread simulates JS."""
    ids_before = set(cs.events.keys())

    def js_side():
        _wait_for(lambda: set(cs.events.keys()) - ids_before)
        new_id = next(iter(set(cs.events.keys()) - ids_before))
        _post(comm_server.server_port, new_id, "pong")

    threading.Thread(target=js_side, daemon=True).start()

    with patch("IPython.display.display"), patch("IPython.display.Javascript"):
        ret = comm_server.result("RETURN('pong')", timeout_seconds=2.0)

    assert ret == '"pong"'


def test_result_raises_timeout_when_no_response(comm_server):
    with (
        patch("IPython.display.display"),
        patch("IPython.display.Javascript"),
        pytest.raises(TimeoutError),
    ):
        comm_server.result("RETURN(nothing)", timeout_seconds=0.2)


def test_result_cleans_up_event_after_timeout(comm_server):
    """After a timeout, the registered event id is removed from the events dict."""
    ids_before = set(cs.events.keys())

    with (
        patch("IPython.display.display"),
        patch("IPython.display.Javascript"),
        pytest.raises(TimeoutError),
    ):
        comm_server.result("RETURN(x)", timeout_seconds=0.2)

    assert set(cs.events.keys()) == ids_before

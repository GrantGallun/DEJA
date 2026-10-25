"""Persistent transport boundaries without account, network, or model access."""
import importlib
import importlib.util
import json
import queue
import subprocess
import threading
import time

import pytest


SCHEMA = {"type": "object", "properties": {"matches": {"type": "array"},
          "note": {"type": "string"}}, "required": ["matches", "note"], "additionalProperties": False}


class Output:
    def __init__(self):
        self.lines = queue.Queue()
    def readline(self, limit=-1):
        return self.lines.get()
    def close(self):
        pass


class Input:
    def __init__(self, process):
        self.process = process
    def write(self, text):
        message = json.loads(text)
        if self.process.mode == "broken_pipe" and message.get("method") == "turn/start":
            raise BrokenPipeError("sensitive diagnostic must not escape")
        if self.process.mode == "blocked_write" and message.get("method") == "turn/start":
            self.process.turn_received.set()
            self.process.exited.wait(2)
            raise BrokenPipeError("stopped")
        self.process.receive(message)
        return len(text)
    def flush(self):
        pass
    def close(self):
        self.process.finish(0)


class Process:
    def __init__(self, command, mode="normal"):
        self.command, self.mode = command, mode
        self.stdout, self.stderr = Output(), Output()
        self.stdin = Input(self)
        self.exited, self.turn_received = threading.Event(), threading.Event()
        self.messages = []
        self.returncode = None
        self.thread_number = 0
        self.current = None
    def emit(self, value):
        self.stdout.lines.put(json.dumps(value) + "\n")
    def reply(self, message, value):
        self.emit({"id": message["id"], "result": value})
    def receive(self, message):
        self.messages.append(message)
        method = message.get("method")
        params = message.get("params", {})
        if method == "initialize":
            self.reply(message, {"userAgent": "fake", "platformFamily": "windows", "platformOs": "windows"})
        elif method == "config/read":
            active = self.mode == "mcp" and 'mcp_servers.node_repl.enabled=false' not in self.command
            servers = {"node_repl": {"enabled": True, "env": {"SECRET": "do-not-print"}}} if active else {}
            self.reply(message, {"config": {"mcp_servers": servers, "web_search": "disabled", "features": {
                "apps": False, "plugins": False, "hooks": False, "shell_tool": False,
                "unified_exec": False, "multi_agent": False, "browser_use": False, "computer_use": False,
                "view_image": False, "image_generation": False, "workspace_dependencies": False},
                "service_tier": "default"}, "origins": {}, "layers": None})
        elif method == "thread/start":
            self.thread_number += 1
            self.reply(message, {"thread": {"id": f"thread-{self.thread_number}"}, "model": params["model"],
                "reasoningEffort": "high" if self.mode == "unsafe_reasoning" else "low",
                "sandbox": {"type": "readOnly"}, "approvalPolicy": "on-request" if self.mode == "unsafe_approval" else "never",
                "instructionSources": [], "serviceTier": params.get("serviceTier")})
        elif method == "thread/unsubscribe":
            self.reply(message, {})
        elif method == "mcpServerStatus/list":
            entries = [{"name": "node_repl", "runtimeStatus": "disabled", "tools": {}, "resources": [],
                        "resourceTemplates": [], "serverInfo": None}] if self.mode == "mcp" else []
            self.reply(message, {"data": entries, "nextCursor": None})
        elif method == "turn/start":
            self.turn_received.set()
            self.current = (params["threadId"], f"turn-{self.thread_number}", params["input"][-1]["text"])
            if self.mode == "early_complete":
                self.complete()
            self.reply(message, {"turn": {"id": self.current[1], "status": "inProgress", "items": []}})
            if self.mode == "early_complete": return
            if self.mode in ("timeout", "hold"):
                return
            if self.mode == "protocol":
                self.stdout.lines.put("PRIVATE_AUTH_DATA\n")
                return
            if self.mode == "approval":
                self.emit({"id": 900, "method": "item/commandExecution/requestApproval", "params": {"command": "PRIVATE"}})
                return
            if self.mode == "tool":
                self.emit({"method": "item/started", "params": {"threadId": self.current[0], "turnId": self.current[1],
                    "item": {"id": "tool", "type": "mcpToolCall"}}})
                return
            if self.mode == "reroute":
                self.emit({"method": "model/rerouted", "params": {"fromModel": "gpt-6-luna", "toModel": "other"}})
                return
            self.complete()
    def complete(self):
        thread_id, turn_id, prompt = self.current
        text = json.dumps({"matches": [], "note": prompt})
        if self.mode == "invalid_json": text = "not json"
        if self.mode == "invalid_result": text = '{"matches": [], "note": 5}'
        if self.mode == "nan": text = '{"matches": [NaN], "note": "bad"}'
        self.emit({"method": "item/completed", "params": {"threadId": thread_id, "turnId": turn_id,
            "item": {"id": "answer", "type": "agentMessage", "phase": "final_answer", "text": text}}})
        self.emit({"method": "turn/completed", "params": {"threadId": thread_id,
            "turn": {"id": turn_id, "status": "failed" if self.mode == "failed" else "completed", "items": []}}})
    def poll(self):
        return self.returncode
    def finish(self, code):
        if self.returncode is None:
            self.returncode = code
            self.exited.set()
            self.stdout.lines.put("")
            self.stderr.lines.put("")
    def terminate(self):
        self.finish(-1)
    def kill(self):
        self.finish(-9)
    def wait(self, timeout=None):
        if not self.exited.wait(timeout): raise subprocess.TimeoutExpired(self.command, timeout)
        return self.returncode


class Factory:
    def __init__(self, *modes):
        self.modes = list(modes) or ["normal"]
        self.processes = []
    def __call__(self, command, **kwargs):
        process = Process(command, self.modes[min(len(self.processes), len(self.modes)-1)])
        self.processes.append(process)
        return process


def client(tmp_path, factory):
    assert importlib.util.find_spec("smartreplay.demo.codex_transport") is not None, "Persistent transport is missing"
    module = importlib.import_module("smartreplay.demo.codex_transport")
    return module.CodexTransport("codex.exe", tmp_path, process_factory=factory)


def test_reuses_child_but_isolates_questions_and_forces_standard_readonly_settings(tmp_path):
    factory = Factory()
    transport = client(tmp_path, factory)
    assert not factory.processes, "Construction must not start an account or process"
    image = tmp_path / "frame.jpg"; image.write_bytes(b"fixture")
    try:
        assert transport.request("first", [image], SCHEMA, model="gpt-6-luna", timeout=1) == {"matches": [], "note": "first"}
        assert transport.request("second", [image], SCHEMA, model="gpt-6-luna", timeout=1) == {"matches": [], "note": "second"}
        assert len(factory.processes) == 1
        process = factory.processes[0]
        threads = [m["params"] for m in process.messages if m.get("method") == "thread/start"]
        turns = [m["params"] for m in process.messages if m.get("method") == "turn/start"]
        assert len(threads) == 2 and all(m["ephemeral"] and m["sandbox"] == "read-only" for m in threads)
        assert all(m["serviceTier"] == "default" for m in threads)
        assert turns[0]["threadId"] != turns[1]["threadId"]
        assert all(m["effort"] == "low" and m["serviceTierForTurn"] == "default" for m in turns)
        assert all(m["approvalPolicy"] == "never" and m["sandboxPolicy"] == {"type": "readOnly", "networkAccess": False} for m in turns)
        assert turns[0]["input"] == [{"type": "localImage", "path": str(image.resolve())}, {"type": "text", "text": "first"}]
        assert turns[0]["outputSchema"] == SCHEMA
        assert [m["params"]["threadId"] for m in process.messages if m.get("method") == "thread/unsubscribe"] == ["thread-1", "thread-2"]
        for setting in ("features.apps=false", "features.plugins=false", "features.hooks=false", "mcp_servers={}", 'service_tier="default"'):
            assert setting in process.command
    finally:
        transport.close()
    assert factory.processes[0].poll() is not None


def test_discovers_names_only_and_disables_recursively_merged_mcp_before_thread_start(tmp_path):
    factory = Factory("mcp", "mcp")
    transport = client(tmp_path, factory)
    try:
        assert transport.request("safe", [], SCHEMA, timeout=1)["note"] == "safe"
        assert len(factory.processes) == 2
        assert factory.processes[0].poll() is not None
        assert not any(m.get("method") in ("thread/start", "turn/start") for m in factory.processes[0].messages)
        assert 'mcp_servers.node_repl.enabled=false' in factory.processes[1].command
    finally:
        transport.close()


@pytest.mark.parametrize("mode", ["broken_pipe", "protocol", "invalid_json", "invalid_result", "nan", "failed", "approval", "tool", "reroute", "unsafe_approval", "unsafe_reasoning"])
def test_bad_connections_fail_without_leaking_diagnostics_and_restart_cleanly(tmp_path, mode):
    factory = Factory(mode, "normal")
    transport = client(tmp_path, factory)
    try:
        with pytest.raises(RuntimeError) as caught:
            transport.request("first", [], SCHEMA, timeout=.5)
        assert "PRIVATE" not in str(caught.value) and "sensitive" not in str(caught.value)
        assert factory.processes[0].poll() is not None
        assert transport.request("retry", [], SCHEMA, timeout=1)["note"] == "retry"
        assert len(factory.processes) == 2
        if mode == "approval":
            assert any(m.get("id") == 900 and m.get("result") == {"decision": "cancel"} for m in factory.processes[0].messages)
    finally:
        transport.close()


@pytest.mark.parametrize("mode", ["timeout", "blocked_write"])
def test_deadline_covers_waiting_and_blocked_pipe_writes(tmp_path, mode):
    factory = Factory(mode, "normal")
    transport = client(tmp_path, factory)
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="timed out|timeout"):
            transport.request("slow", [], SCHEMA, timeout=.05)
        assert time.monotonic()-started < .4
        assert factory.processes[0].poll() is not None
        assert transport.request("retry", [], SCHEMA, timeout=1)["note"] == "retry"
    finally:
        transport.close()


def test_serialized_admission_timeout_does_not_cancel_the_active_question(tmp_path):
    factory = Factory("hold")
    transport = client(tmp_path, factory)
    answers = []
    worker = threading.Thread(target=lambda: answers.append(transport.request("active", [], SCHEMA, timeout=1)))
    worker.start()
    try:
        deadline = time.monotonic()+1
        while not factory.processes and time.monotonic()<deadline: time.sleep(.001)
        assert factory.processes[0].turn_received.wait(1)
        with pytest.raises(RuntimeError, match="timed out|timeout"):
            transport.request("queued", [], SCHEMA, timeout=.02)
        assert factory.processes[0].poll() is None
        factory.processes[0].complete()
        worker.join(1)
        assert answers == [{"matches": [], "note": "active"}]
    finally:
        transport.close(); worker.join(1)


def test_close_cancels_active_request_without_waiting_for_request_lock(tmp_path):
    factory = Factory("hold")
    transport = client(tmp_path, factory)
    errors = []
    def ask():
        try: transport.request("active", [], SCHEMA, timeout=10)
        except RuntimeError as exc: errors.append(exc)
    worker = threading.Thread(target=ask); worker.start()
    deadline = time.monotonic()+1
    while not factory.processes and time.monotonic()<deadline: time.sleep(.001)
    assert factory.processes[0].turn_received.wait(1)
    started = time.monotonic(); transport.close(); transport.close()
    worker.join(.5)
    assert time.monotonic()-started < .5 and not worker.is_alive() and errors
    assert factory.processes[0].poll() is not None
    with pytest.raises(RuntimeError, match="closed"):
        transport.request("late", [], SCHEMA, timeout=1)


def test_turn_notifications_before_rpc_acknowledgement_are_not_lost(tmp_path):
    transport = client(tmp_path, Factory("early_complete"))
    try:
        assert transport.request("early", [], SCHEMA, timeout=.1) == {"matches": [], "note": "early"}
    finally:
        transport.close()


@pytest.mark.parametrize("stop", [False, True])
def test_full_notification_inbox_cannot_hide_connection_failure_or_close(tmp_path, stop):
    module = importlib.import_module("smartreplay.demo.codex_transport")
    transport = client(tmp_path, Factory())
    connection = module._Connection(Process(["fake"]))
    try:
        for _ in range(256): connection.inbox.put_nowait({"method": "thread/status/changed", "params": {}})
        if stop: connection.stop()
        else: connection.fail("Connection failed")
        with pytest.raises(RuntimeError):
            transport._receive(connection, time.monotonic()+.05)
    finally:
        connection.stop(); transport.close()

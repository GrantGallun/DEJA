"""Serialized local Codex app-server client; credentials remain owned by Codex."""
from __future__ import annotations

import json
import math
from pathlib import Path
import queue
import re
import subprocess
import threading
import time

_DISABLED_FEATURES = ("apps", "plugins", "hooks", "shell_tool", "unified_exec", "multi_agent",
                      "browser_use", "computer_use", "view_image", "image_generation", "workspace_dependencies")


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RuntimeError("Codex request timed out. Retry the question.")
    return remaining


def _invalid_constant(_):
    raise ValueError("Non-finite JSON number")


def _valid(value, schema):
    """Check the JSON-schema subset used by our bounded vision results."""
    if isinstance(value, float) and not math.isfinite(value):
        return False
    if schema is False:
        return False
    if schema is True or not isinstance(schema, dict):
        schema = {}
    types = {"object": isinstance(value, dict), "array": isinstance(value, list),
             "string": isinstance(value, str), "boolean": isinstance(value, bool),
             "number": type(value) in (int, float), "integer": type(value) is int, "null": value is None}
    expected = schema.get("type")
    if expected and not any(types.get(t, False) for t in ([expected] if isinstance(expected, str) else expected)):
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if "anyOf" in schema and not any(_valid(value, option) for option in schema["anyOf"]):
        return False
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        if any(key not in value for key in schema.get("required", [])):
            return False
        if schema.get("additionalProperties") is False and any(key not in properties for key in value):
            return False
        return all(_valid(item, properties.get(key, schema.get("additionalProperties", {}))) for key, item in value.items())
    if isinstance(value, list):
        return (len(value) >= schema.get("minItems", 0) and len(value) <= schema.get("maxItems", math.inf)
                and all(_valid(item, schema.get("items", {})) for item in value))
    if isinstance(value, str):
        return schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", math.inf)
    if type(value) in (int, float):
        return schema.get("minimum", -math.inf) <= value <= schema.get("maximum", math.inf)
    return True


class _Connection:
    """Pipe workers keep a blocked read/write outside the caller's deadline."""
    def __init__(self, process):
        self.process = process
        self.inbox, self.outbox = queue.Queue(256), queue.Queue(2)
        self.stopped = threading.Event()
        self.failure = None
        self.sequence = 0
        self.reader = threading.Thread(target=self._read, daemon=True, name="codex-response-reader")
        self.writer = threading.Thread(target=self._write, daemon=True, name="codex-request-writer")
        self.reader.start()
        self.writer.start()

    def fail(self, reason):
        self.failure = reason  # Never store raw server output or auth diagnostics.
        try:
            self.inbox.put_nowait(None)
        except queue.Full:
            pass

    def _read(self):
        try:
            while not self.stopped.is_set():
                line = self.process.stdout.readline(1024 * 1024 + 1)
                if not line:
                    self.fail("Codex connection closed.")
                    return
                if len(line) > 1024 * 1024 or not line.endswith("\n"):
                    self.fail("Invalid Codex protocol response.")
                    return
                message = json.loads(line, parse_constant=_invalid_constant)
                if not isinstance(message, dict):
                    raise ValueError("Invalid envelope")
                self.inbox.put_nowait(message)
        except (OSError, ValueError, TypeError, queue.Full):
            self.fail("Invalid Codex protocol response.")

    def _write(self):
        while not self.stopped.is_set():
            try:
                text, done, errors = self.outbox.get(timeout=.1)
            except queue.Empty:
                continue
            try:
                self.process.stdin.write(text)
                self.process.stdin.flush()
            except (OSError, ValueError):
                errors.append(True)
                self.fail("Codex connection failed.")
            finally:
                done.set()

    def send(self, value, deadline):
        if self.failure or self.stopped.is_set():
            raise RuntimeError(self.failure or "Codex connection closed.")
        done, errors = threading.Event(), []
        text = json.dumps(value, allow_nan=False) + "\n"
        try:
            self.outbox.put_nowait((text, done, errors))
        except queue.Full:
            raise RuntimeError("Codex connection is not accepting requests.") from None
        if not done.wait(_remaining(deadline)):
            raise RuntimeError("Codex request timed out. Retry the question.")
        if errors:
            raise RuntimeError("Codex connection failed.")

    def stop(self, deadline=None):
        self.stopped.set()
        self.fail("Codex connection closed.")
        process = self.process
        try:
            if process.poll() is None:
                process.terminate()
            wait = .1 if deadline is None else max(0, min(.1, deadline-time.monotonic()))
            process.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except OSError:
                pass
        except OSError:
            pass
        # Closing a TextIO pipe while another thread holds its write lock can
        # deadlock. Reap first, then close only after the pipe workers exit.
        def reap():
            try:
                process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                return
            self.reader.join(timeout=.2)
            self.writer.join(timeout=.2)
            if not self.reader.is_alive() and not self.writer.is_alive():
                for stream in (process.stdin, process.stdout):
                    try:
                        stream.close()
                    except (OSError, ValueError):
                        pass
        threading.Thread(target=reap, daemon=True, name="codex-process-reaper").start()


class CodexTransport:
    """One lazy child, fresh ephemeral threads, Standard speed and no tools.

    ``timeout`` includes waiting for another request, startup and inference.
    A failed connection is discarded; the next independent call starts cleanly.
    ``close`` cancels the active child without waiting for the request lock.
    """
    def __init__(self, binary, cwd, *, process_factory=None):
        self.binary, self.cwd = str(binary), Path(cwd).resolve()
        self._factory = process_factory or subprocess.Popen
        self._requests, self._lifecycle = threading.Lock(), threading.Lock()
        self._connection, self._closed = None, False
        self._disabled_mcp = set()

    def _command(self):
        settings = ["mcp_servers={}", 'web_search="disabled"', 'model_reasoning_effort="low"', 'service_tier="default"']
        settings += [f"features.{name}=false" for name in _DISABLED_FEATURES]
        if any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in self._disabled_mcp):
            raise RuntimeError("Codex MCP server names could not be safely disabled.")
        settings += [f"mcp_servers.{name}.enabled=false" for name in sorted(self._disabled_mcp)]
        command = [self.binary, "app-server", "--stdio"]
        for setting in settings:
            command.extend(["-c", setting])
        return command

    def _drop(self, connection, deadline=None):
        with self._lifecycle:
            if self._connection is connection:
                self._connection = None
        connection.stop(deadline)

    def _observe(self, connection, message, deadline):
        method = message.get("method")
        if not method:
            return
        if not isinstance(method, str) or not isinstance(message.get("params", {}), dict):
            raise RuntimeError("Invalid Codex protocol notification.")
        if "id" in message:
            # All server-initiated approvals, tools and credential requests fail
            # closed; their parameters and errors never enter user-visible logs.
            if method in ("item/commandExecution/requestApproval", "item/fileChange/requestApproval"):
                reply = {"result": {"decision": "cancel"}}
            elif method == "item/tool/call":
                reply = {"result": {"success": False, "contentItems": [
                    {"type": "inputText", "text": "Tools are disabled for replay questions."}]}}
            elif method == "mcpServer/elicitation/request":
                reply = {"result": {"action": "decline"}}
            else:
                reply = {"error": {"code": -32601, "message": "Unsupported client request"}}
            connection.send({"id": message["id"], **reply}, deadline)
            raise RuntimeError("Codex requested an unavailable tool or permission.")
        if method in ("model/rerouted", "thread/modelRerouted") or method.startswith("mcpServer/"):
            raise RuntimeError("Codex attempted to enable an unexpected model or connection.")
        if method in ("item/started", "item/completed"):
            kind = message["params"].get("item", {}).get("type")
            if kind not in ("userMessage", "agentMessage", "reasoning"):
                raise RuntimeError("Codex attempted to use a tool for a replay question.")

    def _receive(self, connection, deadline):
        if connection.failure or connection.stopped.is_set():
            raise RuntimeError(connection.failure or "Codex connection closed.")
        try:
            message = connection.inbox.get(timeout=_remaining(deadline))
        except queue.Empty:
            raise RuntimeError("Codex request timed out. Retry the question.") from None
        if message is None or connection.failure or connection.stopped.is_set():
            raise RuntimeError(connection.failure or "Codex connection closed.")
        self._observe(connection, message, deadline)
        return message

    def _rpc(self, connection, method, params, deadline, notifications=None):
        connection.sequence += 1
        request_id = connection.sequence
        connection.send({"id": request_id, "method": method, "params": params}, deadline)
        while True:
            message = self._receive(connection, deadline)
            if "method" in message:
                if notifications is not None:
                    notifications.append(message)
                continue
            if message.get("id") != request_id or ("result" in message) == ("error" in message):
                raise RuntimeError("Invalid Codex protocol response.")
            if "error" in message:
                raise RuntimeError("Codex could not complete the replay request.")
            if not isinstance(message["result"], dict):
                raise RuntimeError("Invalid Codex protocol result.")
            return message["result"]

    def _config_mcp(self, connection, deadline):
        result = self._rpc(connection, "config/read", {"includeLayers": False, "cwd": str(self.cwd)}, deadline)
        # Inspect names and enabled flags only. Never log, retain or inspect
        # server env, URLs, headers, provider credentials or configuration layers.
        config = result.get("config", {})
        features = config.get("features", {})
        if any(features.get(name) is not False for name in _DISABLED_FEATURES):
            raise RuntimeError("Codex tool restrictions could not be verified.")
        if config.get("service_tier") != "default":
            raise RuntimeError("Codex Standard speed could not be verified.")
        servers = config.get("mcp_servers", {})
        if not isinstance(servers, dict) or config.get("web_search") != "disabled":
            raise RuntimeError("Codex connection restrictions could not be verified.")
        self._disabled_mcp.update(servers)
        return {name for name, server in servers.items() if not isinstance(server, dict) or server.get("enabled", True) is not False}

    def _connect(self, deadline):
        for attempt in range(2):
            with self._lifecycle:
                if self._closed:
                    raise RuntimeError("This Codex connection is closed.")
                connection = self._connection
            if connection and (connection.failure or connection.process.poll() is not None):
                self._drop(connection, deadline)
                connection = None
            if connection is None:
                _remaining(deadline)
                process = self._factory(self._command(), cwd=self.cwd, stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                    encoding="utf-8", errors="strict", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                connection = _Connection(process)
                with self._lifecycle:
                    if self._closed:
                        connection.stop(deadline)
                        raise RuntimeError("This Codex connection is closed.")
                    self._connection = connection
                self._rpc(connection, "initialize", {"clientInfo": {"name": "smartreplay", "version": "0.1.0"},
                    "capabilities": {"experimentalApi": True, "explicitGatewayOauth": True, "requestAttestation": False}}, deadline)
                connection.send({"method": "initialized", "params": {}}, deadline)
            active = self._config_mcp(connection, deadline)
            if not active:
                return connection
            self._disabled_mcp.update(active)
            self._drop(connection, deadline)
        raise RuntimeError("Codex external connections could not be disabled.")

    def request(self, prompt, image_paths, schema, model="gpt-6-luna", timeout=60):
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("A positive finite Codex timeout is required.")
        deadline = time.monotonic() + timeout
        if not self._requests.acquire(timeout=_remaining(deadline)):
            raise RuntimeError("Codex request timed out waiting for the current question.")
        connection = None
        try:
            if not isinstance(prompt, str) or not prompt.strip() or not isinstance(model, str) or not model.strip():
                raise ValueError("A model and nonempty replay question are required.")
            images = [str(Path(path).resolve(strict=True)) for path in image_paths]
            connection = self._connect(deadline)
            result = self._rpc(connection, "thread/start", {"model": model, "cwd": str(self.cwd),
                "ephemeral": True, "sandbox": "read-only", "approvalPolicy": "never", "serviceTier": "default"}, deadline)
            if (result.get("model") != model or result.get("sandbox", {}).get("type") != "readOnly"
                    or result.get("approvalPolicy") != "never"
                    or ("reasoningEffort" in result and result["reasoningEffort"] != "low")
                    or result.get("serviceTier") not in (None, "default")):
                raise RuntimeError("Codex did not select the requested model and permissions.")
            thread_id = result["thread"]["id"]
            inventory = self._rpc(connection, "mcpServerStatus/list", {"threadId": thread_id, "limit": 100,
                "detail": "toolsAndAuthOnly"}, deadline)
            entries = inventory.get("data")
            if (not isinstance(entries, list) or inventory.get("nextCursor") or any(
                    not isinstance(entry, dict) or entry.get("name") not in self._disabled_mcp
                    or entry.get("runtimeStatus") != "disabled" or entry.get("tools")
                    or entry.get("resources") or entry.get("resourceTemplates")
                    or entry.get("serverInfo") is not None for entry in entries)):
                raise RuntimeError("Codex external tools remain available.")
            notifications = []
            turn = self._rpc(connection, "turn/start", {"threadId": thread_id,
                "input": [{"type": "localImage", "path": path} for path in images] + [{"type": "text", "text": prompt}],
                "model": model, "effort": "low", "serviceTierForTurn": "default", "approvalPolicy": "never",
                "sandboxPolicy": {"type": "readOnly", "networkAccess": False}, "outputSchema": schema}, deadline, notifications)
            turn_id, answer = turn["turn"]["id"], None
            while True:
                message = notifications.pop(0) if notifications else self._receive(connection, deadline)
                params = message.get("params", {})
                if params.get("threadId") != thread_id:
                    continue
                if message.get("method") == "item/completed" and params.get("turnId") == turn_id:
                    item = params.get("item", {})
                    if item.get("type") == "agentMessage":
                        answer = item.get("text")
                if message.get("method") == "turn/completed" and params.get("turn", {}).get("id") == turn_id:
                    completed = params["turn"]
                    if completed.get("status") != "completed":
                        raise RuntimeError("Codex did not complete the replay question.")
                    if answer is None:
                        answer = next((item.get("text") for item in completed.get("items", []) if item.get("type") == "agentMessage"), None)
                    value = json.loads(answer, parse_constant=_invalid_constant)
                    if not isinstance(value, dict) or not _valid(value, schema):
                        raise RuntimeError("Codex returned an invalid replay result.")
                    try:
                        self._rpc(connection, "thread/unsubscribe", {"threadId": thread_id}, deadline)
                    except RuntimeError:
                        # The answer is already valid; retire the child instead
                        # of retaining an image-bearing thread if cleanup fails.
                        self._drop(connection, deadline)
                    return value
        except Exception as exc:
            # _connect may fail before returning its newly created connection.
            with self._lifecycle:
                failed = connection or self._connection
            if failed:
                self._drop(failed, deadline)
            if isinstance(exc, (RuntimeError, ValueError)) and not isinstance(exc, json.JSONDecodeError):
                if isinstance(exc, RuntimeError):
                    raise
            raise RuntimeError("Codex returned an invalid response or its connection failed.") from None
        finally:
            self._requests.release()

    def close(self):
        with self._lifecycle:
            self._closed = True
            connection, self._connection = self._connection, None
        if connection:
            connection.stop()

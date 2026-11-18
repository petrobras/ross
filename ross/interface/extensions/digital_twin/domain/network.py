# -*- coding: utf-8 -*-
"""Whether this computer is on the Petrobras network.

The equipment models and the PI Data server are reachable only from inside
the company network. Every Digital Twin route that reads a model or process
data asks `on_corporate_network()` first (`api/digital_twin.py`), so a user
outside it gets one clear refusal instead of a DNS or timeout error from deep
inside a model.

The test opens a TCP connection to an internal host. `DT_NETWORK_PROBE` lists
the hosts as `host:port`, separated by commas; one answering is enough. There
is no host built in: an internal address does not belong in a public
repository, and a wrong guess would answer "outside" to everyone. Without the
variable the check fails closed -- the pre-established models stay
unavailable, and `refusal()` says what to set. `DT_NETWORK_CHECK=off` skips
the test, for development with the sample process data.

This check decides what the interface offers. It does not protect the model
files: those are protected by living outside the ROSS repository, in the
Digital Twin app that only Petrobras users can check out (see `dt_app.py`).
"""

import os
import socket
import threading
import time

TIMEOUT = 3.0
CACHE_SECONDS = 60.0

OUTSIDE = (
    "The pre-established rotor models are available only on the Petrobras "
    "network. Connect to the network (or VPN) and try again."
)

UNSET = (
    "The pre-established rotor models are not configured on this computer: "
    "set DT_NETWORK_PROBE to the host:port of an internal server, so the "
    "interface can tell whether it is on the Petrobras network."
)

_cache = {"at": None, "answer": None}
_lock = threading.Lock()


def probes():
    """Return the `(host, port)` pairs to try, empty when none is configured."""
    raw = os.environ.get("DT_NETWORK_PROBE", "").strip()
    if not raw:
        return []
    pairs = []
    for item in raw.split(","):
        host, _, port = item.strip().rpartition(":")
        if host and port.isdigit():
            pairs.append((host, int(port)))
    if not pairs:
        raise ValueError("DT_NETWORK_PROBE has no host:port: %r" % raw)
    return pairs


def check_disabled():
    return os.environ.get("DT_NETWORK_CHECK", "").strip().lower() in {
        "0",
        "off",
        "false",
        "no",
    }


def reachable(host, port, timeout=TIMEOUT):
    """Tell whether a TCP connection to `host:port` opens."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def on_corporate_network(now=None):
    """Tell whether one of the probe hosts answers, remembering it for a minute."""
    if check_disabled():
        return True
    now = time.monotonic() if now is None else now
    with _lock:
        if _cache["at"] is not None and now - _cache["at"] < CACHE_SECONDS:
            return _cache["answer"]
        hosts = probes()
        answer = bool(hosts) and any(reachable(host, port) for host, port in hosts)
        _cache.update(at=now, answer=answer)
        return answer


def refusal():
    """Return the message for a refused request: unset probe, or outside."""
    return OUTSIDE if probes() else UNSET


def forget():
    """Drop the remembered answer, so the next call tests again."""
    with _lock:
        _cache.update(at=None, answer=None)

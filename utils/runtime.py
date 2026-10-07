"""Local bot supervision; no network requests or provider imports."""
from __future__ import annotations

import atexit
import fcntl
import json
import logging
import logging.handlers
import os
from pathlib import Path
import re
import socket
import sys
import threading
import time
from urllib.parse import urlsplit

STOP = threading.Event()
_LOCKS = []


def acquire_instance(path):
    """Exclusive host-local lock, retained across the bot's os.execv updates."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    key = 'BOT_LOCK_FD_' + str(path).encode().hex()
    inherited = os.environ.get(key)
    if inherited:
        try:
            fd = int(inherited)
            stat = os.fstat(fd)
            actual = path.stat()
            if (stat.st_dev, stat.st_ino) == (actual.st_dev, actual.st_ino):
                # The inherited descriptor retains the kernel flock, not just a PID.
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd
        except (OSError, ValueError):
            pass
    handle = path.open('a+')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f'Another bot owns {path}') from None
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    os.set_inheritable(handle.fileno(), True)
    os.environ[key] = str(handle.fileno())
    _LOCKS.append(handle)
    return handle.fileno()


class Health:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.RLock()

    def read(self):
        try:
            data = json.loads(self.path.read_text())
            if not isinstance(data, dict):
                raise ValueError('Health state must be an object')
            return data
        except FileNotFoundError:
            return {}
        # A corrupt scheduling/delivery ledger must never become a fresh post.

    def update(self, **fields):
        with self.lock:
            data = self.read()
            data.update(fields)
            data.update(pid=os.getpid(), updated_at=time.time())
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix('.tmp')
            with tmp.open('w') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            tmp.replace(self.path)
            return data

    def delivery(self, channel, status, **fields):
        with self.lock:
            channels = self.read().get('channels', {})
            previous = channels.get(channel, {})
            channels[channel] = {**previous, **fields, 'status': status,
                                 'checked_at': time.time()}
            if status == 'ok':
                channels[channel]['last_success_at'] = time.time()
            self.update(channels=channels)


def http_status(exc):
    code = getattr(exc, 'status_code', None)
    if code is None:
        code = getattr(getattr(exc, 'response', None), 'status_code', None)
    return code


def category(exc):
    code = http_status(exc)
    if code == 402:
        return 'billing'
    if code in (401, 403):
        return 'authentication'
    if code == 429:
        return 'rate_limit'
    if code and code >= 500:
        return 'provider'
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return 'connectivity'
    return 'pipeline'


def terminal_http_error(exc):
    code = http_status(exc)
    return code is not None and 400 <= code < 500 and code not in (408, 425, 429)


def wait_ready(stop=STOP, timeout=120):
    """Wait for configured HTTP proxies; never print credentials or call AI."""
    import shutil
    import errno
    minimum = int(os.environ.get("BOT_MIN_FREE_BYTES", 2 * 1024**3))
    if shutil.disk_usage(".").free < minimum:
        raise OSError(errno.ENOSPC, "Insufficient disk headroom; generation held")
    endpoints = set()
    for key in ('HTTPS_PROXY', 'HTTP_PROXY', 'ALL_PROXY',
                'https_proxy', 'http_proxy', 'all_proxy'):
        value = os.environ.get(key)
        if value:
            u = urlsplit(value)
            if not u.hostname:
                raise ValueError('Malformed proxy configuration')
            endpoints.add((u.hostname, u.port or (443 if u.scheme == 'https' else 80)))
    deadline = time.monotonic() + timeout
    while not stop.is_set():
        try:
            for endpoint in endpoints:
                with socket.create_connection(endpoint, timeout=2):
                    pass
            return True
        except OSError:
            if time.monotonic() >= deadline:
                raise ConnectionError('Configured proxy is not ready') from None
            stop.wait(2)
    return False


class ConsoleLog:
    """Line-oriented, bounded unattended output; discard animated CR updates."""
    encoding = 'utf-8'
    errors = 'replace'
    _ansi = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')

    def __init__(self, path, max_bytes=10 * 1024 * 1024, backups=3):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=backups, encoding='utf-8')
        self.handler.setFormatter(logging.Formatter('%(message)s'))
        self.buffers = {}
        self.lock = threading.RLock()
        atexit.register(self.flush)

    def isatty(self):
        return False

    def write(self, text):
        with self.lock:
            if '\r' in text:
                return len(text)
            tid = threading.get_ident()
            buf = self.buffers.get(tid, '') + self._ansi.sub('', text)
            while '\n' in buf:
                line, buf = buf.split('\n', 1)
                self.handler.emit(logging.LogRecord('console', logging.INFO, '', 0,
                                                    line, (), None))
            self.buffers[tid] = buf
        return len(text)

    def flush(self):
        self.handler.flush()

    def reconfigure(self, **kwargs):
        pass


def install_console():
    path = os.environ.get('BOT_CONSOLE_LOG')
    if path:
        stream = ConsoleLog(path)
        sys.stdout = sys.stderr = stream

"""Running engine work off the UI thread.

Every long operation — evaluating a design, fitting metamodels, optimizing —
goes through :func:`run_async`. Doing this work inline would freeze the window,
which matters most precisely when the user wants feedback: during a long
optimization.

The engine knows nothing about Qt, so callbacks it invokes (progress, per
generation updates) arrive on the *worker* thread. Passing them through Qt
signals is what marshals them back to the UI thread, and touching a widget
directly from the worker instead would be a crash waiting to happen.
"""

from __future__ import annotations

import traceback
from typing import Any, Callable

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot


class Worker(QObject):
    """Runs one callable on a thread and reports the outcome by signal."""

    progress = Signal(int, int, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    @Slot()
    def run(self) -> None:
        try:
            result = self._fn(*self._args, **self._kwargs)
        except Exception as exc:  # surfaced in the UI rather than lost to stderr
            self.failed.emit(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}")
        else:
            self.finished.emit(result)


class Task:
    """Keeps a worker and its thread alive for the duration of the work.

    Python would otherwise garbage-collect the ``QThread`` as soon as the
    caller returned, tearing down a running thread mid-operation. Holding this
    handle on the caller is what prevents that.
    """

    def __init__(self, worker: Worker, thread: QThread, relay: QObject | None = None):
        self.worker = worker
        self.thread = thread
        # Held so the relay outlives the task; dropping it would silently
        # disconnect the callbacks.
        self.relay = relay

    def wait(self, timeout_ms: int = 10_000) -> bool:
        try:
            return self.thread.wait(timeout_ms)
        except RuntimeError:
            # The underlying QThread was already deleted, which means it had
            # finished; there is nothing left to wait for.
            return True

    @property
    def is_running(self) -> bool:
        try:
            return self.thread.isRunning()
        except RuntimeError:
            return False


# Tasks still in flight. Qt aborts the process if a QThread is destroyed while
# it is still running, so the application must be able to find outstanding work
# and wait for it before tearing anything down.
_ACTIVE: set[Task] = set()


def active_tasks() -> list[Task]:
    return [task for task in _ACTIVE if task.is_running]


def wait_for_all(timeout_ms: int = 10_000) -> bool:
    """Block until every outstanding task has finished.

    Called during shutdown. Any still-running worker holds a reference to
    widgets that are about to be destroyed, so letting the window go first is
    what turns a clean exit into a crash.
    """
    finished = True
    for task in list(_ACTIVE):
        # Ask the loop to exit before blocking on it. A worker that finished
        # while nothing was pumping events may still be sitting idle in exec().
        try:
            task.thread.quit()
        except RuntimeError:
            continue
        if not task.wait(timeout_ms):
            finished = False
    _ACTIVE.clear()
    return finished


class _Relay(QObject):
    """Delivers a worker's results on the thread that started the task.

    Connecting a signal straight to a plain Python callable gives Qt no
    receiver object, so it cannot tell which thread should run the callback and
    invokes it *directly* — on the worker thread. Every completion handler here
    touches widgets, which is not safe off the GUI thread and deadlocks readily.

    Routing through a QObject created by the caller fixes the affinity: the
    relay belongs to the calling thread, so Qt resolves these to queued
    connections and the callbacks run where the widgets live.
    """

    def __init__(
        self,
        on_finished: Callable[[Any], None],
        on_failed: Callable[[str], None] | None,
        on_progress: Callable[[int, int, str], None] | None,
    ):
        super().__init__()
        self._on_finished = on_finished
        self._on_failed = on_failed
        self._on_progress = on_progress

    @Slot(object)
    def finished(self, result: Any) -> None:
        self._on_finished(result)

    @Slot(str)
    def failed(self, message: str) -> None:
        if self._on_failed is not None:
            self._on_failed(message)

    @Slot(int, int, str)
    def progress(self, current: int, total: int, message: str) -> None:
        if self._on_progress is not None:
            self._on_progress(current, total, message)


def run_async(
    fn: Callable[..., Any],
    on_finished: Callable[[Any], None],
    on_failed: Callable[[str], None] | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
    *args: Any,
    **kwargs: Any,
) -> Task:
    """Run ``fn`` on a worker thread, delivering results on the calling thread."""
    thread = QThread()
    worker = Worker(fn, *args, **kwargs)
    worker.moveToThread(thread)

    relay = _Relay(on_finished, on_failed, on_progress)

    thread.started.connect(worker.run)
    worker.finished.connect(relay.finished)
    worker.failed.connect(relay.failed)
    worker.progress.connect(relay.progress)

    # Quit the thread's event loop on either outcome, then let Qt delete both
    # objects once the thread has actually finished.
    #
    # These two must be DIRECT connections. The QThread object itself lives in
    # the calling thread, so an automatic connection would *queue* quit() back
    # to that thread — and shutdown blocks it inside wait_for_all(), so the quit
    # would never be delivered and the wait would never return. QThread.quit is
    # documented thread-safe, so invoking it from the worker is correct.
    worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
    worker.failed.connect(thread.quit, Qt.ConnectionType.DirectConnection)
    thread.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)

    task = Task(worker, thread, relay)
    _ACTIVE.add(task)
    thread.finished.connect(lambda: _ACTIVE.discard(task))

    thread.start()
    return task


class ProgressReporter(QObject):
    """Adapts an engine progress callback into a Qt signal.

    Engine functions take plain callables and know nothing about Qt. Wrapping
    one of these lets the same callback cross the thread boundary safely.
    """

    progress = Signal(int, int, str)

    def callback(self, current: int, total: int, message: str = "") -> None:
        self.progress.emit(current, total, message)

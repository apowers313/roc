# mypy: disable-error-code="no-untyped-def"

from threading import Thread
from time import sleep
from unittest.mock import MagicMock

import pytest

from roc.breakpoint import breakpoints


def true_fn() -> bool:
    return True


def false_fn() -> bool:
    return False


class TestBreakpoint:
    @pytest.fixture(scope="function", autouse=True)
    def clear_breakpoints(self):
        breakpoints.clear()
        assert breakpoints.count == 0
        breakpoints.resume()
        assert not breakpoints.brk
        sleep(0.0001)  # XXX: fix race condition

    @pytest.fixture(scope="class", autouse=True)
    def breakpoint_thread(self):
        stop_threads = False

        def breakpoint_loop():
            while True:
                breakpoints.check()
                sleep(0.01)
                if stop_threads:
                    break

        t = Thread(target=breakpoint_loop)
        t.start()

        yield

        stop_threads = True
        breakpoints.resume()
        t.join(timeout=15)
        if t.is_alive():
            raise Exception(
                "!!! THREAD IS STILL ALIVE AT END OF BREAKPOINT TESTS. TESTS WILL LIKELY HANG."
            )

    @pytest.fixture(scope="function")
    def mock_break_true(self):
        mm = MagicMock(return_value=True)
        mm.__name__ = "mock_break_true"
        return mm

    @pytest.fixture(scope="function")
    def mock_break_false(self):
        mm = MagicMock(return_value=False)
        mm.__name__ = "mock_break_false"
        return mm

    def test_state(self) -> None:
        assert not breakpoints.brk
        assert breakpoints.state == "running"
        breakpoints.brk = True
        assert breakpoints.state == "stopped"
        breakpoints.brk = False

    def test_add(self, mock_break_true) -> None:
        num = breakpoints.count

        breakpoints.add(mock_break_true)

        assert breakpoints.count == num + 1
        assert "mock_break_true" in breakpoints

    def test_add_with_name(self, mock_break_true) -> None:
        num = breakpoints.count

        breakpoints.add(mock_break_true, name="foo")

        assert breakpoints.count == num + 1
        assert "foo" in breakpoints

    def test_remove(self, mock_break_true) -> None:
        breakpoints.add(mock_break_true, name="foo")
        num = breakpoints.count
        breakpoints.remove("foo")
        assert breakpoints.count == num - 1
        assert "foo" not in breakpoints

    def test_check_trigger(self, mock_break_true) -> None:
        assert mock_break_true.call_count == 0
        breakpoints.add(mock_break_true, name="foo")
        assert breakpoints.state == "running"

        sleep(0.2)

        assert breakpoints.state == "stopped"
        assert breakpoints.trigger == "foo"
        assert mock_break_true.call_count == 1

    def test_check_no_trigger(self, mock_break_false) -> None:
        assert mock_break_false.call_count == 0
        breakpoints.add(mock_break_false, name="foo")
        assert breakpoints.state == "running"
        breakpoints.check()
        assert breakpoints.state == "running"
        assert mock_break_false.call_count == 1

    def test_list_empty(self, mock_break_true) -> None:
        lst = str(breakpoints)
        assert lst == "0 breakpoint(s). State: running."

    def test_list_one(self, mock_break_true) -> None:
        breakpoints.add(false_fn, name="foo")
        sleep(0.2)
        lst = str(breakpoints)
        assert lst.startswith(
            "1 breakpoint(s). State: running.\n\n"
            "         Breakpoints    Source\n"
            "--  ---  -------------  ---------------------\n"
            " 0       foo            breakpoint_test.py:"
        )

    def test_list_stopped(self, mock_break_true) -> None:
        breakpoints.add(true_fn, name="foo")

        sleep(0.2)

        lst = str(breakpoints)
        assert lst.startswith(
            "1 breakpoint(s). State: stopped.\n\n"
            "         Breakpoints    Source\n"
            "--  ---  -------------  ---------------------\n"
            " 0  *    foo            breakpoint_test.py:"
        )

    def test_resume(self, mock_break_true) -> None:
        breakpoints.add(mock_break_true, name="foo")

        sleep(0.2)

        assert breakpoints.state == "stopped"

        breakpoints.resume()

        assert breakpoints.state == "running"
        assert breakpoints.trigger is None

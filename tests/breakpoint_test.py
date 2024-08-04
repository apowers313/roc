# mypy: disable-error-code="no-untyped-def"


from unittest.mock import MagicMock

import pytest

from roc.breakpoint import breakpoints


def true_fn() -> bool:
    return True


class TestBreakpoint:
    @pytest.fixture(scope="function", autouse=True)
    def clear_breakpoints(self):
        breakpoints.clear()
        assert breakpoints.count == 0
        breakpoints.resume()
        assert not breakpoints.brk

    @pytest.fixture(scope="class")
    def mock_break_true(self):
        return MagicMock(return_value=True)

    @pytest.fixture(scope="class")
    def mock_break_false(self):
        return MagicMock(return_value=False)

    def test_state(self) -> None:
        assert not breakpoints.brk
        assert breakpoints.state == "running"
        breakpoints.brk = True
        assert breakpoints.state == "stopped"

    def test_add(self, mock_break_true) -> None:
        num = breakpoints.count

        breakpoints.add("foo", mock_break_true)

        assert breakpoints.count == num + 1
        assert "foo" in breakpoints

    def test_remove(self, mock_break_true) -> None:
        breakpoints.add("foo", mock_break_true)
        num = breakpoints.count
        breakpoints.remove("foo")
        assert breakpoints.count == num - 1
        assert "foo" not in breakpoints

    def test_check_trigger(self, mock_break_true) -> None:
        assert mock_break_true.call_count == 0
        breakpoints.add("foo", mock_break_true)
        assert breakpoints.state == "running"
        breakpoints.check()
        assert breakpoints.state == "stopped"
        assert breakpoints.trigger == "foo"
        assert mock_break_true.call_count == 1

    def test_check_no_trigger(self, mock_break_false) -> None:
        assert mock_break_false.call_count == 0
        breakpoints.add("foo", mock_break_false)
        assert breakpoints.state == "running"
        breakpoints.check()
        assert breakpoints.state == "running"
        assert mock_break_false.call_count == 1

    def test_list_empty(self, mock_break_true) -> None:
        lst = str(breakpoints)
        assert lst == "0 breakpoint(s). State: running."

    def test_list_one(self, mock_break_true) -> None:
        breakpoints.add("foo", true_fn)
        lst = str(breakpoints)
        assert lst == (
            "1 breakpoint(s). State: running.\n\n"
            "         Breakpoints    File\n"
            "--  ---  -------------  ---------------------\n"
            " 0       foo            breakpoint_test.py:11"
        )

    def test_list_stopped(self, mock_break_true) -> None:
        breakpoints.add("foo", true_fn)
        breakpoints.check()
        lst = str(breakpoints)
        assert lst == (
            "1 breakpoint(s). State: stopped.\n\n"
            "         Breakpoints    File\n"
            "--  ---  -------------  ---------------------\n"
            " 0  *    foo            breakpoint_test.py:11"
        )

    def test_resume(self, mock_break_true) -> None:
        breakpoints.add("foo", mock_break_true)
        breakpoints.check()
        assert breakpoints.state == "stopped"

        breakpoints.resume()

        assert breakpoints.state == "running"
        assert breakpoints.trigger is None

# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/breakpoint.py."""

import pytest

from roc.game.breakpoint import Breakpoint, BreakpointInfo, _breakpoints_dict


@pytest.fixture(autouse=True)
def clear_breakpoints():
    """Clear global breakpoints before and after each test."""
    _breakpoints_dict.clear()
    yield
    _breakpoints_dict.clear()


class TestBreakpointInfo:
    def test_dataclass(self):
        fn = lambda: True
        info = BreakpointInfo(fn=fn, src="test.py:1")
        assert info.fn is fn
        assert info.src == "test.py:1"

    def test_dataclass_none_src(self):
        fn = lambda: True
        info = BreakpointInfo(fn=fn, src=None)
        assert info.src is None


class TestBreakpointConstructor:
    def test_init(self):
        bp = Breakpoint()
        assert bp.brk is False
        assert bp.trigger is None
        assert bp.lock is not None


class TestBreakpointLen:
    def test_empty(self):
        bp = Breakpoint()
        assert len(bp) == 0

    def test_after_add(self):
        bp = Breakpoint()
        bp.add(lambda: True, name="test")
        assert len(bp) == 1


class TestBreakpointContains:
    def test_not_contains(self):
        bp = Breakpoint()
        assert "test" not in bp

    def test_contains(self):
        bp = Breakpoint()
        bp.add(lambda: True, name="test")
        assert "test" in bp


class TestBreakpointStr:
    def test_empty(self):
        bp = Breakpoint()
        result = str(bp)
        assert "0 breakpoint(s)" in result
        assert "running" in result

    def test_with_breakpoints(self):
        bp = Breakpoint()

        def my_fn():
            return True

        bp.add(my_fn, name="test_bp", src="test.py:10")
        result = str(bp)
        assert "1 breakpoint(s)" in result
        assert "test_bp" in result
        assert "test.py:10" in result


class TestBreakpointCount:
    def test_count(self):
        bp = Breakpoint()
        assert bp.count == 0
        bp.add(lambda: True, name="a")
        assert bp.count == 1
        bp.add(lambda: True, name="b")
        assert bp.count == 2


class TestBreakpointState:
    def test_running(self):
        bp = Breakpoint()
        assert bp.state == "running"

    def test_stopped(self):
        bp = Breakpoint()
        bp.brk = True
        assert bp.state == "stopped"


class TestBreakpointAdd:
    def test_add_with_name(self):
        bp = Breakpoint()
        fn = lambda: True
        bp.add(fn, name="my_bp")
        assert "my_bp" in bp
        assert _breakpoints_dict["my_bp"].fn is fn

    def test_add_without_name_uses_dunder_name(self):
        bp = Breakpoint()

        def my_condition():
            return True

        bp.add(my_condition)
        assert "my_condition" in bp

    def test_add_without_name_or_dunder_name(self):
        bp = Breakpoint()

        # Create a callable without __name__
        class NoNameCallable:
            def __call__(self):
                return True

        fn = NoNameCallable()
        assert not hasattr(fn, "__name__")
        bp.add(fn)
        assert "<unknown>" in bp

    def test_add_overwrite(self):
        bp = Breakpoint()
        fn1 = lambda: True
        fn2 = lambda: False
        bp.add(fn1, name="test")
        bp.add(fn2, name="test", overwrite=True)
        assert _breakpoints_dict["test"].fn is fn2

    def test_add_duplicate_raises(self):
        bp = Breakpoint()
        bp.add(lambda: True, name="test")
        with pytest.raises(Exception, match="already exists"):
            bp.add(lambda: True, name="test")

    def test_add_with_src(self):
        bp = Breakpoint()
        bp.add(lambda: True, name="test", src="custom_source")
        assert _breakpoints_dict["test"].src == "custom_source"


class TestBreakpointRemove:
    def test_remove_existing(self):
        bp = Breakpoint()
        bp.add(lambda: True, name="test")
        bp.remove("test")
        assert "test" not in bp

    def test_remove_nonexisting_raises(self):
        bp = Breakpoint()
        with pytest.raises(Exception, match="doesn't exist"):
            bp.remove("nonexistent")


class TestBreakpointClear:
    def test_clear(self):
        bp = Breakpoint()
        bp.add(lambda: True, name="a")
        bp.add(lambda: True, name="b")
        assert bp.count == 2
        bp.clear()
        assert bp.count == 0


class TestBreakpointDoBreak:
    def test_do_break_sets_state(self):
        bp = Breakpoint()
        bp.do_break(trigger="test_trigger", quiet=True)
        assert bp.brk is True
        assert bp.trigger == "test_trigger"
        # Clean up - release lock
        bp.lock.release()

    def test_do_break_double_is_noop(self):
        bp = Breakpoint()
        bp.do_break(trigger="first", quiet=True)
        bp.do_break(trigger="second", quiet=True)
        assert bp.trigger == "first"  # unchanged
        # Clean up
        bp.lock.release()

    def test_do_break_default_trigger(self):
        bp = Breakpoint()
        bp.do_break(quiet=True)
        assert bp.trigger == "<user request>"
        bp.lock.release()


class TestBreakpointResume:
    def test_resume_clears_state(self):
        bp = Breakpoint()
        bp.do_break(trigger="test", quiet=True)
        assert bp.brk is True
        bp.resume(quiet=True)
        assert bp.brk is False
        assert bp.trigger is None  # type: ignore[unreachable]

    def test_resume_when_not_broken_is_noop(self):
        bp = Breakpoint()
        # Should not raise
        bp.resume(quiet=True)
        assert bp.brk is False


class TestBreakpointCheck:
    def test_check_triggers_on_true(self):
        bp = Breakpoint()
        bp.add(lambda: True, name="always_break")

        # Run check in a way that we can verify do_break was called
        # Since do_break acquires the lock, we need to release it afterward
        import threading

        def run_check():
            bp.check()

        t = threading.Thread(target=run_check)
        t.start()

        # Give time for the check to trigger the break
        import time

        time.sleep(0.1)
        assert bp.brk is True
        bp.resume(quiet=True)
        t.join(timeout=2)

    def test_check_continues_on_false(self):
        bp = Breakpoint()
        bp.add(lambda: False, name="never_break")
        bp.check()
        assert bp.brk is False


class TestBreakpointList:
    def test_list_prints(self, capsys):
        bp = Breakpoint()
        bp.list()
        captured = capsys.readouterr()
        assert "0 breakpoint(s)" in captured.out

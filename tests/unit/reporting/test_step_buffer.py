# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/reporting/step_buffer.py."""

from roc.reporting.step_buffer import StepBuffer

from roc.reporting.run_store import StepData


def _make_step(step: int, game: int) -> StepData:
    """Create a minimal StepData for testing."""
    return StepData(step=step, game_number=game)


class TestStepBuffer:
    def test_push_and_get_step(self):
        buf = StepBuffer(capacity=10)
        sd = _make_step(1, 1)
        buf.push(sd)
        assert buf.get_step(1) is sd
        assert buf.get_step(2) is None

    def test_get_latest(self):
        buf = StepBuffer(capacity=10)
        assert buf.get_latest() is None
        buf.push(_make_step(1, 1))
        buf.push(_make_step(2, 1))
        latest = buf.get_latest()
        assert latest is not None
        assert latest.step == 2

    def test_min_max_step(self):
        buf = StepBuffer(capacity=10)
        assert buf.min_step == 0
        assert buf.max_step == 0
        buf.push(_make_step(5, 1))
        buf.push(_make_step(10, 1))
        assert buf.min_step == 5
        assert buf.max_step == 10

    def test_game_numbers(self):
        buf = StepBuffer(capacity=10)
        buf.push(_make_step(1, 1))
        buf.push(_make_step(2, 1))
        buf.push(_make_step(3, 2))
        assert buf.game_numbers == [1, 2]

    def test_steps_per_game(self):
        """Regression: games endpoint showed (0 steps) for all games."""
        buf = StepBuffer(capacity=100)
        for i in range(1, 6):
            buf.push(_make_step(i, 1))
        for i in range(6, 9):
            buf.push(_make_step(i, 2))
        counts = buf.steps_per_game()
        assert counts == {1: 5, 2: 3}

    def test_step_range_for_game(self):
        """Regression: step-range endpoint ignored game filter for live run."""
        buf = StepBuffer(capacity=100)
        for i in range(1, 11):
            buf.push(_make_step(i, 1))
        for i in range(11, 16):
            buf.push(_make_step(i, 2))

        assert buf.step_range_for_game(1) == (1, 10)
        assert buf.step_range_for_game(2) == (11, 15)
        assert buf.step_range_for_game(99) == (0, 0)

    def test_capacity_eviction(self):
        buf = StepBuffer(capacity=3)
        buf.push(_make_step(1, 1))
        buf.push(_make_step(2, 1))
        buf.push(_make_step(3, 1))
        buf.push(_make_step(4, 1))
        assert len(buf) == 3
        assert buf.min_step == 2
        assert buf.get_step(1) is None

    def test_listener_called_on_push(self):
        buf = StepBuffer(capacity=10)
        calls = []
        buf.add_listener(lambda: calls.append(1))
        buf.push(_make_step(1, 1))
        assert calls == [1]

    def test_remove_listener(self):
        buf = StepBuffer(capacity=10)
        calls = []
        fn = lambda: calls.append(1)
        buf.add_listener(fn)
        buf.push(_make_step(1, 1))
        buf.remove_listener(fn)
        buf.push(_make_step(2, 1))
        assert calls == [1]  # only one call, not two

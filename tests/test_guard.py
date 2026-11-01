from highnoon.core.guard import ModuleGuard


def explode():
    raise ValueError("boom")


def test_returns_the_result_when_nothing_fails():
    assert ModuleGuard().call("game", lambda a: a + 1, 1) == 2


def test_a_failure_returns_the_fallback_and_is_counted():
    guard = ModuleGuard()
    assert guard.call("game", explode, fallback=[]) == []
    assert guard.failures == {"game": 1}


def test_a_repeated_error_prints_once(capsys):
    guard = ModuleGuard()
    for _ in range(3):
        guard.call("game", explode)
    assert capsys.readouterr().out.count("game failed") == 1
    assert guard.failures == {"game": 3}

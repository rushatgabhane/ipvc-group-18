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


def test_an_injected_fault_fails_only_that_module():
    from highnoon.core.guard import parse_faults

    guard = ModuleGuard(parse_faults(["hand"]))
    assert guard.call("hand", lambda: 1, fallback=0) == 0
    assert guard.call("game", lambda: 1, fallback=0) == 1
    assert guard.failures == {"hand": 1}


def test_a_partial_fault_fails_about_that_share_of_calls():
    from highnoon.core.guard import parse_faults

    guard = ModuleGuard(parse_faults(["face:0.3"]), seed=1)
    results = [guard.call("face", lambda: 1, fallback=0) for _ in range(1000)]
    assert 250 < results.count(0) < 350

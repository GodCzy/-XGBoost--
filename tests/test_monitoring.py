from monitoring import ProcessMonitor


def dummy_optimizer(objective, bounds):
    return [1.0, 2.0], objective([1.0, 2.0])


def dummy_objective(params):
    return sum(params)


def test_process_monitor_triggers_optimizer():
    monitor = ProcessMonitor(10, dummy_optimizer, dummy_objective, [(0, 1), (0, 1)])
    params, val = monitor.adjust(20)
    assert params == [1.0, 2.0]
    assert val == 3.0
    params2, val2 = monitor.adjust(5)
    assert params2 is None
    assert val2 == 5

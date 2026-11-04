from experiments.babilong_linked_selection.prepare_long_rollout_job import CPU_TESTS
from experiments.babilong_linked_selection.submit_long_rollout_job import finished_component, submit, valid_cpu_check


def test_component_gate_requires_all_200_cases_on_both_models():
    assert finished_component({"all_complete": True, "completed": {"qwen": 200, "gpt": 200}})
    assert not finished_component({"all_complete": True, "completed": {"qwen": 200, "gpt": 199}})
    assert not finished_component({"all_complete": False, "completed": {"qwen": 200, "gpt": 200}})


def test_cpu_validation_gate_requires_the_exact_current_suite_and_no_api_calls():
    status = {"cpu_tests_passed": True, "tests": CPU_TESTS, "passed_count": len(CPU_TESTS),
              "submit_performed": False, "online_api_calls": 0}
    assert valid_cpu_check(status)
    assert not valid_cpu_check(dict(status, tests=CPU_TESTS[:-1]))
    assert not valid_cpu_check(dict(status, online_api_calls=1))
    assert not valid_cpu_check(dict(status, passed_count=1))


def test_does_not_touch_scheduler_until_both_component_studies_finish(tmp_path):
    root = tmp_path / "query_cartridges_linked_roll32_dev_20261104"
    assert submit(tmp_path, root) is None
    assert not root.exists()

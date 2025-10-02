import pytest

from elasticai.creator.testing.cocotb_pytest import create_name_for_build_test_subdir


def simple_function():
    pass


def test_create_name_for_simple_function():
    actual = create_name_for_build_test_subdir(simple_function)
    expected = "simple_function"
    assert actual == expected


def function_with_args(a, b, c):
    pass


def test_create_name_function_with_args():
    actual = create_name_for_build_test_subdir(function_with_args, 1, 2, c="abcd")
    expected = "function_with_args_a_1_b_2_c_abcd"
    assert actual == expected


def test_create_name_for_function_with_list_args():
    actual = create_name_for_build_test_subdir(
        function_with_args, [1, 2, 3], 2, c="abcd"
    )
    h = hash((1, 2, 3)).to_bytes(length=8, signed=True).hex()
    expected = f"function_with_args_a_{h}_b_2_c_abcd"
    assert actual == expected


def test_create_name_for_fn_with_dict_args():
    actual = create_name_for_build_test_subdir(
        function_with_args, {"a": 1}, 2, c="abcd"
    )
    h = hash((("a", 1),)).to_bytes(length=8, signed=True).hex()
    expected = f"function_with_args_a_{h}_b_2_c_abcd"
    assert actual == expected


@pytest.fixture
def dummy_fixture_fn(request):
    def run(local_namespace):
        print(request.node.callspec)
        namespace_without_fixture = {
            k: v for k, v in local_namespace.items() if k != "dummy_fixture_fn"
        }
        return create_name_for_build_test_subdir(
            request.function, **namespace_without_fixture
        )

    return run


@pytest.mark.parametrize("x", [4])
def test_pass_parameter_to_fixture(dummy_fixture_fn, x):
    actual = dummy_fixture_fn(locals())
    expected = f"test_pass_parameter_to_fixture_x_{x}"
    assert actual == expected

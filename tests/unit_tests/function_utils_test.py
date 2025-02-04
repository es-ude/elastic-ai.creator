from elasticai.creator.function_utils import FunctionDecorator


def test_can_call_with_name_and_fn():
    outer_name = ""

    def callback(name: str, fn):
        nonlocal outer_name
        outer_name = name
        return fn

    decorate = FunctionDecorator(callback)

    def fn():
        return "x"

    fn = decorate("fn_0", fn)
    assert "x" == fn()
    assert outer_name == "fn_0"

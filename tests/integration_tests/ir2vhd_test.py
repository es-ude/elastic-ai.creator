from elasticai.creator.ir2vhdl import Implementation


class TestTranslatingToSkeleton:
    def test_translate(self):
        ir = Implementation(name="my_skeleton", type="skeleton", attributes={})

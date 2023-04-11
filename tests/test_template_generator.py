import re

from elasticai.creator.hdl.vhdl.base_template_generator import BaseTemplateGenerator
from elasticai.creator.hdl.vhdl.manifest import Manifest


def test_generated_template_passes_through_x_signal():
    manifest_content = """[elasticai.creator]
version = "==0.34"

[[elasticai.creator.layers]]
name = "MyLayer"

[elasticai.creator.layers.flavor]
name = "hw_block"
pass_through = ["x"]
"""
    parsed_manifest = Manifest.parse(manifest_content)
    my_layer_manifest = parsed_manifest.layers[0]
    generator = BaseTemplateGenerator(
        name=my_layer_manifest["name"],
        pass_through=my_layer_manifest["flavor"]["pass_through"]
    )
    required_pass_through_line = r"^\s*y <- x;$"
    occurrences = 0
    for line in generator.generate().splitlines():
        if re.match(required_pass_through_line, line):
            occurrences += 1
    assert occurrences == 1





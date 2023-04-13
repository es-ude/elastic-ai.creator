import pytest
from pytest_bdd import given, parsers, scenario, then, when

from elasticai.creator.hdl.vhdl.base_template_generator import BaseTemplateGenerator
from elasticai.creator.hdl.vhdl.manifest import InvalidManifestConfig, Manifest


@given("a manifest content", target_fixture="manifest_lines")
def manifest_lines():
    return """[elasticai.creator]
version = "==0.34"

[elasticai.creator.layer]
""".splitlines()


@given(parsers.parse("it specifies {line}"))
def add_line(manifest_lines, line):
    return manifest_lines.append(line)


@given("i parse the manifest content", target_fixture="parsed_manifest")
def parsed_manifest(manifest_lines):
    return Manifest.parse("\n".join(manifest_lines))


@when("i generate the template", target_fixture="template")
def template(parsed_manifest):
    generator = BaseTemplateGenerator(
        pass_through=parsed_manifest.layer["pass_through"],
    )
    return generator.generate().splitlines()


@then(parsers.parse("it contains the line {line}"))
def contains_lines(template, line):
    lines = line.split(" and ")
    assert template[22 : 22 + len(lines)] == lines


@scenario("features/generate_template.feature", "Pass through x input signal")
def test_pass_through_x():
    pass


@scenario("features/generate_template.feature", "Pass through y_address signal")
def test_pass_through_y_address():
    pass


@scenario("features/generate_template.feature", "Pass through y_address and enable")
def test_pass_through_y_address_and_enable():
    pass


@scenario("features/generate_template.feature", "Parsing incorrect pass_through values")
def test_parsing_incorrect_pass_through_values():
    pass


@then("parsing the manifest throws an error")
def parsing_manifest_throws_error(manifest_lines):
    with pytest.raises(InvalidManifestConfig):
        Manifest.parse("\n".join(manifest_lines))

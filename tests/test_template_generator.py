import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from elasticai.creator.hdl.vhdl.base_template_generator import BaseTemplateGenerator
from elasticai.creator.hdl.vhdl.manifest import parse

scenarios("features/generate_template_from_manifest.feature")


@given(
    parsers.parse("a manifest with pass_through = {pass_through_signals}"),
    target_fixture="manifest_with_pass_through",
)
def manifest_with_pass_through(pass_through_signals: str):
    lines = manifest_lines()
    lines.append(
        "pass_through = {}".format(pass_through_signals),
    )
    parsed = parsed_manifest(lines)
    return parsed


@when("generating the template", target_fixture="template")
def template(manifest_with_pass_through):
    generator = BaseTemplateGenerator(
        pass_through=manifest_with_pass_through.layer["pass_through"],
    )
    return generator.generate().splitlines()


@then(parsers.parse("it contains the lines: {line}"))
@then(parsers.parse("it contains the line: {line}"))
def it_contains(line: str, template):
    lines = line.split(" and ")
    template = template[22:]
    assert template[: len(lines)] == lines


@then("generating the template raises a ValueError")
def generating_template_raises_error(manifest_with_pass_through):
    with pytest.raises(
        ValueError,
        match="found: \w+(, \w+)*, expected one or more of enable, x, y_address",
    ):
        template(manifest_with_pass_through)


def parsed_manifest(manifest_lines):
    return parse("\n".join(manifest_lines))


def manifest_lines():
    return """[elasticai.creator]
version = "==0.34"

[elasticai.creator.layer]
""".splitlines()

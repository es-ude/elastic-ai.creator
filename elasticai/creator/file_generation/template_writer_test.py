from .template import SimpleTemplate
from .template_writer import TemplateWriter
from .virtual_path import VirtualPath


def test_template_writer():
    template = SimpleTemplate(parameters={}, content=["a", "b"])
    build_dir = VirtualPath("root")
    file_path = build_dir.create_subpath("myfile.txt")
    with file_path.open() as f:
        writer = TemplateWriter(f)
        writer.write(template)
    expected = "a\nb\n"
    assert expected == file_path.read()

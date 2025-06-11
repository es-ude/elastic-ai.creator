from elasticai.creator.graph.name_generation import NameRegistry


def test_can_generate_unique_name():
    reg = NameRegistry()
    reg.prepopulate(["a", "b", "c"])
    assert reg.get_unique_name("a") == "a_1"
    assert reg.get_unique_name("a") == "a_2"


def test_name_is_not_altered_if_already_unique():
    reg = NameRegistry()
    reg.prepopulate(["a", "b", "c"])
    assert reg.get_unique_name("d") == "d"


def test_prepopulated_suffixes_are_counted():
    reg = NameRegistry()
    reg.prepopulate(["a", "a_1", "a_2"])
    assert reg.get_unique_name("a") == "a_3"


def test_start_counting_from_highest_suffix():
    reg = NameRegistry()
    reg.prepopulate(["a_3"])
    assert reg.get_unique_name("a") == "a_4"


def test_count_either_no_suffix_or_zero_as_one():
    reg = NameRegistry()
    reg.prepopulate(["a", "a_0"])
    assert reg.get_unique_name("a") == "a_1"


def test_use_maximum_of_suffixes():
    reg = NameRegistry()
    reg.prepopulate(["a", "a_3", "a_2"])

    assert reg.get_unique_name("a") == "a_4"

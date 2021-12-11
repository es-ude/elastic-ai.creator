from typing import Any

from torch.nn import Module


def tag(module: Module, **new_tags: Any) -> Module:
    """Add tags to any object wrapping it in a TagWrapper if necessary

    new_tags will override possibly existing tags
    """
    old_tags = get_tags(module)
    tags = old_tags | new_tags
    module._elasticai_tags = tags

    def get_tags_method(self=module):
        return self._elasticai_tags

    module.elasticai_tags = get_tags_method
    return module


def get_tags(module: Module) -> dict:
    if hasattr(module, 'elasticai_tags'):
        return module.elasticai_tags()
    return {}


def has_tag(module: Module, tag_name) -> bool:
    return tag_name in get_tags(module)

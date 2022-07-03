from elasticai.creator.resource_utils import PathType
from elasticai.creator.vhdl.vhdl_component import VHDLComponent


class CustomTemplateMapping:
    def __init__(self):
        self._mapping = dict()

    @staticmethod
    def _get_cls_name(component_cls: type[VHDLComponent]) -> str:
        return f"{component_cls.__module__}.{component_cls.__name__}"

    def add(self, component_cls: type[VHDLComponent], template_path: PathType) -> None:
        self._mapping.update({self._get_cls_name(component_cls): template_path})

    def get(self, component_cls: type[VHDLComponent]) -> PathType | None:
        if self._get_cls_name(component_cls) in self._mapping:
            return self._mapping[self._get_cls_name(component_cls)]
        return None

    def __str__(self) -> str:
        return str(self._mapping)

from abc import ABC, abstractmethod

from elasticai.creator.vhdl.designs.folder import Folder
from elasticai.creator.vhdl.hardware_description_language import design


class Design(design.Design, ABC):
    @abstractmethod
    def save_to(self, destination: Folder):
        ...

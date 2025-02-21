import re


class NameRegistry:
    def __init__(self):
        self._registry = {}

    def prepopulate(self, names):
        for name in names:
            match = re.match(r"(.+)_(\d+)$", name)
            if match:
                name = match.group(1)
                suffix = int(match.group(2))
            else:
                suffix = 0
            self._registry[name] = self._registry.get(name, suffix) + 1
        return self

    def get_unique_name(self, name):
        if name not in self._registry:
            self._registry[name] = 0
            return name

        new_name = f"{name}_{self._registry[name]}"
        self._registry[name] += 1
        return new_name

import re


class NameRegistry:
    def __init__(self):
        self._registry = {}

    def _get_name_count(self, name):
        return self._registry.get(name, 0)

    def prepopulate(self, names):
        for name in names:
            match = re.match(r"(.+)_(\d+)$", name)
            suffix = 0
            if match:
                name = match.group(1)
                suffix = int(match.group(2))
            suffix = max(suffix, self._get_name_count(name))
            self._registry[name] = suffix
        return self

    def get_unique_name(self, name):
        if name not in self._registry:
            self._registry[name] = 0
            return name

        new_name = f"{name}_{self._registry[name] + 1}"
        self._registry[name] += 1
        return new_name

def generate_default_suffix(default_value: None | str) -> str:
    if default_value is None:
        return ""
    else:
        return f" := {default_value}"

def parse(text):
    separated = text.split(":")
    fields = ("source", "line", "column", "time", "type", "content")
    parsed = dict(zip(fields, separated))
    parsed["type"] = parsed["type"][1:-1]
    parsed["time"] = parsed["time"][1:]
    parsed["line"] = int(parsed["line"])
    parsed["column"] = int(parsed["column"])
    return parsed

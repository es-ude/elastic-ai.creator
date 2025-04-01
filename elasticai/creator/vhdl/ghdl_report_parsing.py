def parse_report(text: str):
    lines = text.split("\n")[:-2]

    def split_first_five_colons(line):
        all_split = line.split(":")
        reassembled = all_split[0:5] + [":".join(all_split[5:])]
        return reassembled

    separated = [split_first_five_colons(line) for line in lines]
    fields = ("source", "line", "column", "time", "type", "content")

    def parse_line(line):
        parsed = dict(zip(fields, line))
        parsed["type"] = parsed["type"][1:-1]
        parsed["time"] = parsed["time"][1:]
        parsed["line"] = int(parsed["line"])
        parsed["column"] = int(parsed["column"])
        return parsed

    return [parse_line(line) for line in separated]

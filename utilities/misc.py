def pull_text(src, coding="utf-8-sig"):
    with open(src, mode="r", encoding=coding) as opensource:
        return opensource.read()


def isnumber(string: str):
    if not string:
        return False
    s = string[1:] if string[0] == "-" and "-" not in string[1:] else string
    if s.isdigit() or s.isnumeric():
        return True
    s = s.replace(",", ".")
    if "." not in s or s.count(".") > 1:
        return False
    if all(part.isdigit() for part in s.split(".")):
        return True
    return False

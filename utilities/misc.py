"""Pure Python and Python stdlib based utilities are here.
This module aims to be PyPy compatible."""
from parser.reparse import reparse_txt


def dehungarize(src, outflpath=None, incoding=None, outcoding=None, **reparse_kw):

    hun_to_asc = {"á": "a", "é": "e", "í": "i",
                  "ó": "o", "ö": "o", "ő": "o",
                  "ú": "u", "ü": "u", "ű": "u",
                  "Á": "A", "É": "E", "Í": "I",
                  "Ó": "O", "Ö": "O", "Ő": "O",
                  "Ú": "U", "Ü": "U", "Ű": "U"}

    if ("/" in src or "\\" in src) and len(src) < 200:
        src = pull_text(src, coding=incoding)
    src = "".join(char if char not in hun_to_asc else hun_to_asc[char] for char in src)
    if reparse_kw:
        src = reparse_txt(src, **reparse_kw)
    if outflpath is None:
        return src
    else:
        with open(outflpath, "w", encoding=outcoding) as outfl:
            outfl.write(src)
            outfl.close()


def pull_text(src, coding="utf-8-sig", **reparse_kw):
    with open(src, mode="r", encoding=coding) as opensource:
        src = opensource.read()
    return reparse_txt(src, **reparse_kw) if reparse_kw else src


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

def enumprogress(data, prefix="", postfix=""):
    total = len(data)
    strln = len(str(total))
    for i, element in enumerate(data, start=1):
        print(f"\r{prefix}{i:>{strln}}/{total}{postfix}", end="")
        yield element
    print()


def percentprogress(data, prefix="", postfix="", precision=2):
    total = len(data)
    strln = 4+precision+1
    for i, element in enumerate(data, start=1):
        print(f"\r{prefix}{i/total:>{strln}.{precision}%}{postfix}", end="")
        yield element
    print()


def itemprogress(data, prefix="", postfix="", printlast=False):
    strpostfix = f" - {data[-1]}" if printlast else ""
    for element in data:
        print(f"\r{prefix}{element}{strpostfix}")

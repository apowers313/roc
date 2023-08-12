import re


def normalize_whitespace(s: str) -> str:
    s = s.strip().replace("\n", " ")
    return re.sub(r"\s+", " ", s)


def assert_similar(expected: str, actual: str, changes: list[tuple[str, str]]) -> None:
    match_str = re.escape(expected)
    for change in changes:
        # print(f"replacing '{change[0]}' with '{change[1]}'")
        match_str = match_str.replace(change[0], change[1])
    assert re.search(match_str, actual), f"expected '{expected}' to be similar to '{actual}'"

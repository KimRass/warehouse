import re
from typing import Literal


def insert_en_dash(text, mode: Literal["start_of_line", "end_of_line"]):
    """
    Hyphen: -, Minus sign: −, En dash: –, Em dash: —
    """
    assert mode in ["start_of_line", "end_of_line"],\
        """The argument `mode` must be one of `"start_of_line"`, `"end_of_line"`."""
    if mode == "start_of_line":
        new_text = re.sub(pattern=r"([^+-/~])(\n)", repl=lambda x: x.group(1) + "–" + x.group(2), string=text)
    elif mode == "end_of_line":
        new_text = re.sub(
            pattern=r"([^+-/~]+)(\n)([^+-/~]+)", repl=lambda x: "–" + x.group(2) + x.group(3), string=text
        )
    return new_text

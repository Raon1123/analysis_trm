"""Line-aware YAML frontmatter / YAML-file loader.

Every value can be mapped back to the 1-based file line that declares it, because every trace edge
needs a `{file, line}` source. Uses yaml.compose (node marks) next to yaml.safe_load.
"""
from __future__ import annotations

import yaml

OPENED: list[str] = []  # every file whose *content* was read (audit for the locked/withheld test)


class ParseError(Exception):
    def __init__(self, path, msg):
        super().__init__(f"{path}: {msg}")
        self.path = str(path)
        self.msg = msg


def read_text(path) -> str:
    OPENED.append(str(path))
    with open(path, encoding="utf-8") as fh:
        return fh.read()


class FM:
    """Parsed mapping plus the line of each key / list item."""

    def __init__(self, path, data, root, base):
        self.path = str(path)
        self.data = data if isinstance(data, dict) else {}
        self._root = root
        self._base = base  # file line of yaml line index 0

    def line(self, *keys, default=1) -> int:
        """1-based file line of the value at the key/index path; the closest known ancestor otherwise."""
        node = self._root
        last = default
        for k in keys:
            if node is None:
                break
            if isinstance(node, yaml.MappingNode):
                nxt = None
                for kn, vn in node.value:
                    if kn.value == k:
                        nxt = (kn, vn)
                        break
                if nxt is None:
                    break
                last = nxt[0].start_mark.line + self._base
                node = nxt[1]
            elif isinstance(node, yaml.SequenceNode) and isinstance(k, int) and k < len(node.value):
                node = node.value[k]
                last = node.start_mark.line + self._base
            else:
                break
        return last


def _compose(path, text, base):
    try:
        data = yaml.safe_load(text)
        root = yaml.compose(text)
    except yaml.YAMLError as e:  # pragma: no cover - message text only
        raise ParseError(path, f"YAML error: {str(e).splitlines()[0] if str(e) else e}")
    if data is not None and not isinstance(data, dict):
        raise ParseError(path, "top-level YAML is not a mapping")
    return FM(path, data or {}, root, base)


def load_frontmatter(path, required=True):
    """`---` delimited frontmatter of a markdown file. Missing frontmatter -> ParseError if required else None."""
    text = read_text(path)
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        if required:
            raise ParseError(path, "no frontmatter (first line is not '---')")
        return None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return _compose(path, "\n".join(lines[1:i]), 2)
    if required:
        raise ParseError(path, "frontmatter is not closed by '---'")
    return None


def load_yaml_file(path):
    return _compose(path, read_text(path), 1)

"""Label vocabulary and discovery of the raw captures.

Raw data is only ever *read*. Labels come from the dataset's own naming convention
(Experiment-2: class folder names; Experiment-3: file-name prefix). They are used as targets only,
never as model inputs.
"""
from __future__ import annotations

from pathlib import Path

from .utils import natural_key

ROOMS = ("A", "B")

# Global (union) activity vocabulary, alphabetical so that indices are stable.
ACTIVITIES = ["empty", "fall", "lying", "sit", "sitdown", "stand", "standup", "walk"]

# Activities that exist in each room.
ROOM_ACTIVITIES = {
    "A": ["fall", "sit", "sitdown", "stand", "standup", "walk"],
    "B": ["empty", "lying", "sit", "stand", "walk"],
}
SHARED_ACTIVITIES = ["sit", "stand", "walk"]
# Activities observed only in one room ("target-exclusive" when that room is the target).
EXCLUSIVE_ACTIVITIES = {"A": ["fall", "sitdown", "standup"], "B": ["empty", "lying"]}

ROOM_A_FOLDERS = {
    "fall": "fall", "sit": "sit", "sitdown": "sitdown",
    "stand": "stand", "standup": "standup", "walk": "walk",
}
ROOM_B_PREFIXES = {
    "Empty": "empty", "Lying": "lying", "Sitting": "sit", "Standing": "stand", "Walking": "walk",
}


def other_room(room: str) -> str:
    return "B" if room == "A" else "A"


def is_junk(path: Path) -> bool:
    """macOS archive housekeeping (__MACOSX/, ._*, .DS_Store) must never be treated as data."""
    path = Path(path)
    return "__MACOSX" in path.parts or path.name.startswith("._") or path.name == ".DS_Store"


def _named_dirs(root: Path, name: str):
    if root.name == name and root.is_dir():
        yield root
    for p in root.rglob(name):
        if p.is_dir() and not is_junk(p):
            yield p


def find_room_a_root(raw_root) -> Path:
    """Locate Experiment-2's 'csidata' directory at any depth below raw_root."""
    raw_root = Path(raw_root)
    cands = []
    for p in _named_dirs(raw_root, "csidata"):
        if any((p / folder).is_dir() for folder in ROOM_A_FOLDERS):
            cands.append(p)
    if not cands:
        raise FileNotFoundError(
            f"Could not find Experiment-2 'csidata/' (with fall/sit/... sub-folders) under {raw_root}. "
            "Check paths.raw_root in configs/experiment.yaml.")
    cands.sort(key=lambda p: (len(p.parts), str(p)))
    return cands[0]


def find_room_b_root(raw_root) -> Path:
    """Locate Experiment-3's 'Data' directory (flat folder of *.csv) at any depth below raw_root."""
    raw_root = Path(raw_root)
    cands = []
    for p in _named_dirs(raw_root, "Data"):
        n = sum(1 for f in p.glob("*.csv")
                if not is_junk(f) and f.stem.split("_")[0] in ROOM_B_PREFIXES)
        if n > 0:
            cands.append((len(p.parts), -n, str(p)))
    if not cands:
        raise FileNotFoundError(
            f"Could not find Experiment-3 'Data/' (with Empty*/Lying*/... .csv files) under {raw_root}. "
            "Check paths.raw_root in configs/experiment.yaml.")
    cands.sort()
    return Path(cands[0][2])


def list_room_a_captures(csidata: Path) -> list:
    """Room A: class folders = development captures, csidata/validation/<class>/ = supplied validation."""
    csidata = Path(csidata)
    caps = []
    for split, base in (("dev", csidata), ("val", csidata / "validation")):
        if not base.is_dir():
            continue
        for folder, label in ROOM_A_FOLDERS.items():
            d = base / folder
            if not d.is_dir():
                continue
            files = [f for f in d.glob("*.pcap") if not is_junk(f)]
            for f in sorted(files, key=lambda p: natural_key(p.name)):
                caps.append({"room": "A", "split": split, "label": label,
                             "stem": f.stem, "file_name": f.name, "path": f})
    return caps


def list_room_b_captures(data_dir: Path) -> list:
    """Room B: flat folder of CSV captures; the activity is the file-name prefix."""
    data_dir = Path(data_dir)
    caps = []
    files = [f for f in data_dir.glob("*.csv") if not is_junk(f)]
    for f in sorted(files, key=lambda p: natural_key(p.name)):
        prefix = f.stem.split("_")[0]
        if prefix not in ROOM_B_PREFIXES:
            continue
        caps.append({"room": "B", "split": None, "label": ROOM_B_PREFIXES[prefix],
                     "stem": f.stem, "file_name": f.name, "path": f})
    return caps


def make_capture_id(room: str, split: str, label: str, stem: str) -> str:
    """Stable internal capture identity (used only for grouping / provenance, never as a feature)."""
    return f"{room}|{split}|{label}|{stem}"

"""
ColorSAR - train / val / test split by scene

Patches cut from one Sentinel scene look alike, so a model that trained on some of
them has effectively seen the rest. Whole scenes are therefore held out: every
patch of a scene lands in the same split, and each terrain contributes at least one
scene to val and to test.

A file name such as ROIs1868_summer_s1_59_p10.png belongs to scene
"ROIs1868_summer_59". A scene that appears under two terrain folders is assigned
by the terrain holding most of its patches, and all of its patches move together.

For each terrain the held-out scenes are the subset whose patch count is closest to
the requested fraction. The choice depends only on the file names, so it is the
same on every machine and for every model.

Limit: the data holds one summer scene, so no split can test a season the model
has not seen.

    python splits.py --root v_2      # print the scenes of each split
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np

TEST_FRAC = 0.10
VAL_FRAC = 0.10

_S1_NAME = re.compile(r"^(?P<roi>.+)_s1_(?P<scene>\d+)_p\d+\.png$")


def scene_of(pair):
    """(sar_path, optical_path, terrain) -> scene id, e.g. 'ROIs1868_summer_59'."""
    name = os.path.basename(pair[0])
    m = _S1_NAME.match(name)
    if m is None:
        raise ValueError(f"cannot read a scene id from {name!r}; "
                         "expected <roi>_s1_<scene>_p<patch>.png")
    return f"{m['roi']}_{m['scene']}"


def _closest_subset(sizes, target):
    """Names of the non-empty subset of `sizes` (name -> count) whose total is closest to target."""
    reach = {0: ()}
    for name in sorted(sizes):
        for total, chosen in list(reach.items()):
            reach.setdefault(total + sizes[name], chosen + (name,))
    del reach[0]
    return reach[min(reach, key=lambda total: (abs(total - target), total))]


def scene_split(pairs):
    """-> {"train": [...], "val": [...], "test": [...]} lists of pairs, whole scenes each."""
    by_scene = defaultdict(list)
    for p in pairs:
        by_scene[scene_of(p)].append(p)

    by_terrain = defaultdict(dict)
    for scene, members in by_scene.items():
        counts = Counter(p[2] for p in members)
        primary = min(counts, key=lambda t: (-counts[t], t))
        by_terrain[primary][scene] = len(members)

    where = {}
    for terrain, sizes in sorted(by_terrain.items()):
        if len(sizes) < 3:
            raise ValueError(f"terrain {terrain!r} has {len(sizes)} scene(s); holding out whole "
                             "scenes for val and test needs at least 3")
        total = sum(sizes.values())
        rest = dict(sizes)
        for name, frac in (("test", TEST_FRAC), ("val", VAL_FRAC)):
            spare = len(rest) - (2 if name == "test" else 1)
            smallest_first = dict(sorted(rest.items(), key=lambda kv: (kv[1], kv[0])))
            chosen = _closest_subset(rest, frac * total)
            if len(chosen) > spare:
                chosen = tuple(smallest_first)[:spare]
            for scene in chosen:
                where[scene] = name
                del rest[scene]

    out = {"train": [], "val": [], "test": []}
    for scene, members in sorted(by_scene.items()):
        out[where.get(scene, "train")].extend(members)
    return out


def scenes(pairs):
    return sorted({scene_of(p) for p in pairs})


def pair_key(pair):
    """(sar_path, optical_path, terrain) -> 'terrain/<s1 filename>'."""
    return f"{pair[2]}/{os.path.basename(pair[0])}"


def split_record(split):
    """Scenes of each part of a split (stored in checkpoints and split.json)."""
    return {name: scenes(part) for name, part in split.items()}


def write_split(path, split, used=None):
    """split.json: the scenes of each part, plus the exact pairs used (`used`: name -> pairs)."""
    data = {"rule": "scene split (splits.scene_split)", "scenes": split_record(split)}
    for name, part in (used or {}).items():
        data[name] = sorted(pair_key(p) for p in part)
    with open(path, "w") as f:
        json.dump(data, f, indent=0)


def subsample(pairs, n, seed=42):
    """Seeded random subset of size n (all of them when n <= 0 or n >= len)."""
    if n <= 0 or n >= len(pairs):
        return list(pairs)
    idx = np.random.RandomState(seed).permutation(len(pairs))[:n]
    return [pairs[i] for i in sorted(idx)]


if __name__ == "__main__":
    from color_utils import find_pairs

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="v_2")
    args = ap.parse_args()

    pairs = find_pairs(args.root)
    if not pairs:
        raise SystemExit(f"No s1/s2 pairs found under {args.root}")
    for name, part in scene_split(pairs).items():
        print(f"\n{name}: {len(part)} pairs ({100 * len(part) / len(pairs):.1f}%), "
              f"{len(scenes(part))} scenes")
        table = Counter((p[2], scene_of(p)) for p in part)
        for (terrain, scene), n in sorted(table.items()):
            print(f"  {terrain:<12}{scene:<28}{n:>6}")

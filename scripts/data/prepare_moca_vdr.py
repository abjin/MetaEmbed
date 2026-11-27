"""Convert MoCa-CL-Pairs VDR subsets into the layout MetaEmbed's training loader expects.

Input (downloaded from HF `moca-embed/MoCa-CL-Pairs`):
    <raw_dir>/<subset>/<subset>.parquet         columns: qry, qry_image_path, pos_text,
                                                pos_image_path, neg_text, neg_image_path
    <raw_dir>/images/...                         extracted supplement_images_zip/*.tar.gz

Output (read by `InterleavedDataset` in colpali_engine/utils/mm_dataset_transformation.py via
`load_dataset($DATA_DIR + "MoCa_train_with_image", subset, split="original")`):
    <out_dir>/MoCa_train_with_image/README.md                      config/split declarations
    <out_dir>/MoCa_train_with_image/<subset>/original-*.parquet    columns: qry, qry_image,
                                                pos_text, pos_image, neg_text, neg_image_0

Images are stored as `datasets.Image()` with absolute paths only (no bytes), so the parquet
files stay small and decode lazily from <raw_dir>/images.

Usage:
    python scripts/data/prepare_moca_vdr.py \
        --raw-dir /home/gijin/sample-adaptive-embedding/data_raw/moca \
        --out-dir /home/gijin/sample-adaptive-embedding/data_dir \
        --subsets tevatron_colpali visrag_ind
"""

import argparse
import ast
import os

import pyarrow.parquet as pq
from datasets import Dataset, Features, Image, Sequence, Value

DS_NAME = "MoCa_train_with_image"
FEATURES = Features(
    {
        "qry": Value("string"),
        "qry_image": Image(),
        "pos_text": Value("string"),
        "pos_image": Image(),
        "neg_text": Sequence(Value("string")),
        "neg_image_0": Image(),
    }
)


def parse_list(value):
    # MoCa stores negative lists as their Python repr, e.g. "['a', 'b']"
    if isinstance(value, list):
        return value
    if value is None or value == "":
        return []
    parsed = ast.literal_eval(value)
    return parsed if isinstance(parsed, list) else [parsed]


def to_image(raw_dir, rel_path, missing):
    if not rel_path:
        return None
    path = os.path.join(raw_dir, rel_path)
    if not os.path.isfile(path):
        missing.append(path)
        return False
    return {"path": path, "bytes": None}


def convert_subset(raw_dir, out_dir, subset, rows_per_file):
    rows = pq.read_table(os.path.join(raw_dir, subset, f"{subset}.parquet")).to_pylist()
    out = {k: [] for k in FEATURES}
    missing, dropped = [], 0
    for row in rows:
        neg_texts = parse_list(row["neg_text"])
        neg_paths = parse_list(row["neg_image_path"])
        qry_image = to_image(raw_dir, row["qry_image_path"], missing)
        pos_image = to_image(raw_dir, row["pos_image_path"], missing)
        neg_image_0 = to_image(raw_dir, neg_paths[0], missing) if neg_paths else None
        if False in (qry_image, pos_image, neg_image_0) or not neg_texts:
            dropped += 1
            continue
        out["qry"].append(row["qry"])
        out["qry_image"].append(qry_image)
        out["pos_text"].append(row["pos_text"])
        out["pos_image"].append(pos_image)
        out["neg_text"].append(neg_texts)
        out["neg_image_0"].append(neg_image_0)

    ds = Dataset.from_dict(out, features=FEATURES)
    subset_dir = os.path.join(out_dir, DS_NAME, subset)
    os.makedirs(subset_dir, exist_ok=True)
    for name in os.listdir(subset_dir):
        if name.startswith("original-") and name.endswith(".parquet"):
            os.remove(os.path.join(subset_dir, name))
    num_files = max(1, -(-len(ds) // rows_per_file))
    for i in range(num_files):
        shard = ds.shard(num_shards=num_files, index=i, contiguous=True)
        shard.to_parquet(
            os.path.join(subset_dir, f"original-{i:05d}-of-{num_files:05d}.parquet")
        )
    print(
        f"{subset}: {len(rows)} rows in, {len(ds)} rows out, {dropped} dropped "
        f"({len(missing)} missing image files{', e.g. ' + missing[0] if missing else ''})"
    )


def write_readme(out_dir, subsets):
    lines = ["---", "configs:"]
    for subset in subsets:
        lines += [
            f"- config_name: {subset}",
            "  data_files:",
            "  - split: original",
            f"    path: {subset}/original-*.parquet",
        ]
    lines += ["---", "", "MoCa-CL-Pairs VDR subsets converted by scripts/data/prepare_moca_vdr.py", ""]
    with open(os.path.join(out_dir, DS_NAME, "README.md"), "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--subsets", nargs="+", default=["tevatron_colpali", "visrag_ind"])
    parser.add_argument("--rows-per-file", type=int, default=20000)
    args = parser.parse_args()

    raw_dir = os.path.abspath(args.raw_dir)
    out_dir = os.path.abspath(args.out_dir)
    for subset in args.subsets:
        convert_subset(raw_dir, out_dir, subset, args.rows_per_file)
    write_readme(out_dir, args.subsets)


if __name__ == "__main__":
    main()

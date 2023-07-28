import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from pycocotools import mask as coco_mask
from typing import Dict


def load_annotation(ann: Dict) -> Dict:
    """Parse hiertext annotation

    Args:
        ann (Dict): hiertext annotation

    Returns:
        Dict: parsed annotation. Keys: image_id, width, height, labels, segments, iscrowds
    """
    labels, segments, iscrowds = [], [], []
    for p in ann["paragraphs"]:
        labels.append("paragraph")
        segments.append(p["vertices"])
        iscrowds.append(0 if p["legible"] else 1)
        for l in p["lines"]:
            labels.append("line")
            segments.append(l["vertices"])
            iscrowds.append(0 if l["legible"] else 1)
            for w in l["words"]:
                labels.append("word")
                segments.append(w["vertices"])
                iscrowds.append(0 if w["legible"] else 1)
    return dict(
        image_id=ann["image_id"],
        width=ann["image_width"],
        height=ann["image_height"],
        labels=labels,
        segments=segments,
        iscrowds=iscrowds,
    )


def convert_rle(sample: Dict) -> Dict:
    """Convert polygon segment annotation to RLE format

    Args:
        sample (Dict): segment annotation. Keys: image_id, width, height, labels, segments, iscrowds

    Returns:
        Dict: RLE annotation. Keys: image_id, height, width, paragraph, line, word
    """
    image_id, height, width = sample["image_id"], sample["height"], sample["width"]
    result = dict(image_id=image_id, height=height, width=width)
    for category in ["paragraph", "line", "word"]:
        polygons = [
            np.array(polygon).flatten().tolist()
            for label, polygon in zip(sample["labels"], sample["segments"])
            if label == category
        ]
        rles = coco_mask.frPyObjects(polygons, height, width)
        rle = coco_mask.merge(rles)
        result[category] = rle["counts"]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for split in ["train", "validation"]:
        anns_path = str(Path(args.dataset_dir) / "gt" / f"{split}.jsonl")
        anns = json.load(open(anns_path, "r"))["annotations"]
        results = list()
        for ann in tqdm(anns):
            sample = load_annotation(ann)
            result = convert_rle(sample)
            results.append(result)
        pd.DataFrame(results).to_csv(
            str(Path(args.output_dir) / f"{split}.csv"), index=False
        )

import re
import torch
from collections import defaultdict

_TILE_RE = re.compile(r"_tile_(\d+)_(\d+)_")

def extract_xy_from_tile_stem(stem: str) -> tuple[int, int]:
    """Extract (x, y) tile coordinates from a tile filename stem."""
    match = _TILE_RE.search(stem)
    if match is None:
        raise ValueError(f"Could not extract coordinates from tile name: {stem}")
    return int(match.group(1)), int(match.group(2))


def group_indices_by_slide(dataset, task: str = "inflammation"):
    """Group a HistologyDataset's per-tile sample indices into MIL bags."""
    key_to_indices = defaultdict(list)
    for i, s in enumerate(dataset.samples):
        key = (s["slide_name"], s["metadata"]["particle_id"]) if task == "tissue" else s["slide_name"]
        key_to_indices[key].append(i)
    return list(key_to_indices.values())


def mil_collate(batch):
    """Collate a batch of MIL bags (with embeddings) into flat tensors plus bag indices."""
    bag_images_list, bag_embeds_list = [], []
    labels_list, bag_idx_list = [], []

    for b, (bag_images, bag_embeds, bag_label) in enumerate(batch):
        m = bag_images.size(0)

        bag_images_list.append(bag_images)
        bag_embeds_list.append(bag_embeds)
        labels_list.append(torch.tensor(bag_label, dtype=torch.float))

        bag_idx_list.append(torch.full((m,), b, dtype=torch.long))

    images_flat = torch.cat(bag_images_list, dim=0)
    embeds_flat = torch.cat(bag_embeds_list, dim=0)
    bag_idx = torch.cat(bag_idx_list, dim=0)
    labels = torch.stack(labels_list, dim=0)

    return images_flat, bag_idx, labels, embeds_flat

def mil_collate_no_embeddings(batch):
    """Collate a batch of MIL bags (without embeddings) into flat tensors plus bag indices."""
    bag_images_list = []
    labels_list, bag_idx_list = [], []

    for b, (bag_images, bag_label) in enumerate(batch):
        m = bag_images.size(0)

        bag_images_list.append(bag_images)
        labels_list.append(torch.tensor(bag_label, dtype=torch.float))

        bag_idx_list.append(torch.full((m,), b, dtype=torch.long))

    images_flat = torch.cat(bag_images_list, dim=0)
    bag_idx = torch.cat(bag_idx_list, dim=0)
    labels = torch.stack(labels_list, dim=0)

    return images_flat, bag_idx, labels
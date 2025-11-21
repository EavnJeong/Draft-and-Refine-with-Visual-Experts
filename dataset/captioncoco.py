import os
import json
from torch.utils.data import Dataset


class COCODatasetCaptioning(Dataset):
    def __init__(self, root_dir, split='train', limit=None):
        assert split in ['train', 'val']
        self.split = split
        self.root_dir = root_dir

        self.image_dir = os.path.join(root_dir, 'images', f'{split}2014')

        annotation_path = os.path.join(root_dir, 'annotations', f'captions_{split}2014.json')
        with open(annotation_path, 'r') as f:
            data = json.load(f)

        self.id_to_filename = {img['id']: img['file_name'] for img in data['images']}
        self.samples = [(ann['image_id'], ann['caption']) for ann in data['annotations']]
        if limit is not None:
            self.samples = self.samples[:limit]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_id, caption = self.samples[idx]
        filename = self.id_to_filename[image_id]
        image_path = os.path.join(self.image_dir, filename)

        info = {
            "image": image_path,
            "caption": caption,
        }

        return info
import json
import os
from torch.utils.data import Dataset


class VQAv2Dataset(Dataset):
    def __init__(self, root_dir, split='train', limit=None):
        assert split in ['train', 'val']
        self.split = split
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images', f'{split}2014')

        question_path = os.path.join(root_dir, 'questions', f'v2_OpenEnded_mscoco_{split}2014_questions.json')
        annotation_path = os.path.join(root_dir, 'annotations', f'v2_mscoco_{split}2014_annotations.json')
        
        with open(question_path, 'r') as f:
            self.questions = json.load(f)['questions']
        
        with open(annotation_path, 'r') as f:
            ann_data = json.load(f)['annotations']
        self.annotations = {ann['question_id']: ann for ann in ann_data}
        if limit is not None:
            self.questions = self.questions[:limit]

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        q = self.questions[idx]
        question_id = q['question_id']
        image_id = q['image_id']
        image_file = f'COCO_{self.split}2014_{image_id:012d}.jpg'
        image_path = os.path.join(self.image_dir, image_file)
        
        question = q['question']
        answer = [a['answer'] for a in self.annotations[question_id]['answers']]

        infos = {
            "image": image_path,
            "question": question,
            "answer": answer,
            "question_id": question_id,
        }
        return infos
def vqav2_collate_fn(batch):
    batch_dict = {
        "images": [item["image"] for item in batch],
        "questions": [item["question"] for item in batch],
        "answers": [item["answer"] for item in batch],
        "question_ids": [item["question_id"] for item in batch],
    }
    return batch_dict


def vizwiz_collate_fn(batch):
    batch_dict = {
        "images": [item["image"] for item in batch],
        "questions": [item["question"] for item in batch],
        "answers": [item["answers"] for item in batch],
        "answer_types": [item["answer_type"] for item in batch],
        "answerables": [item["answerable"] for item in batch],
    }
    return batch_dict


def gqa_collate_fn(batch):
    batch_dict = {
        "question_id": [item["question_id"] for item in batch],
        "images": [item["image"] for item in batch],
        "questions": [item["question"] for item in batch],
        "answers": [item["answer"] for item in batch],
        "image_ids": [item["image_id"] for item in batch],
    }
    return batch_dict


def textvqa_collate_fn(batch):
    return {
        "images": [b["image"] for b in batch],
        "questions": [b["question"] for b in batch],
        "answers": [b["answer"] for b in batch],  # list of 10 answers each
        "question_ids": [b["question_id"] for b in batch],
        "ocr_tokens" : [b["ocr_token"] for b in batch],
    }
    

def ocrvqa_collate_fn(batch):
    return {
        "images": [b["image"] for b in batch],
        "questions": [b["question"] for b in batch],
        "answers": [b["answer"] for b in batch],
        "question_ids": [b["question_id"] for b in batch],
        "ocr_tokens" : [b["ocr_token"] for b in batch],
    }
    

def coco_collate_fn(batch):
    batch_dict = {
        "images": [item["image"] for item in batch],
        "captions": [item["caption"] for item in batch],
    }
    return batch_dict


def nocaps_collate_fn(batch):
    return {
        "images": [b["image"] for b in batch],
        "captions": [b["captions"] for b in batch],
        "image_ids": [b["image_id"] for b in batch],
        "domains": [b["domain"] for b in batch],
    }


def flickr_collate_fn(batch):
    """
    Collate function for Flickr captioning dataset.
    - Each image may have multiple captions.
    - Returns batch dict compatible with CLIP-style evaluation.
    """
    batch_dict = {
        "images": [b["image"] for b in batch],   
        "captions": [b["caption"] for b in batch], 
    }
    return batch_dict


def vcr_collate_fn(batch):
    return {
        "images": [b["image"] for b in batch],
        "questions": [b["question"] for b in batch],
        "answer_choices": [b["answer_choice"] for b in batch],
        "rationales": [b["rationales"] for b in batch],
        "answers": [b["answers_texts"] for b in batch],
        "rationale_texts": [b["rationale_texts"] for b in batch],
        "answer_labels": [b["answer_label"] for b in batch],
        "rationale_labels": [b["rationale_label"] for b in batch],
        "question_ids": [b["question_ids"] for b in batch],
    }


def vsr_collate_fn(batch):
    return {
        "images": [b["image"] for b in batch],
        "questions": [b["question"] for b in batch],
        "answers": [b["answer"] for b in batch],  
    }


def okvqa_collate_fn(batch):
    """
    Collate function for QK-VQA (VQAv2/OK-VQA style datasets)
    Converts a list of samples into a batch dictionary.
    """
    batch_dict = {
        "question_ids": [item["question_id"] for item in batch],
        "image_ids": [item["image_id"] for item in batch],
        "images": [item["image"] for item in batch],
        "questions": [item["question"] for item in batch],
        "answers": [item.get("answers", []) for item in batch],
        "multiple_choice_answer": [
            item.get("multiple_choice_answer", None) for item in batch
        ],
    }
    return batch_dict


def aokvqa_collate_fn(batch):
    """
    Collate function for A-OKVQA (Augmented OK-VQA)
    Supports both direct-answer (DA) and multiple-choice (MC) modes.
    """
    batch_dict = {
        "question_ids": [item["question_id"] for item in batch],
        "image_ids": [item["image_id"] for item in batch],
        "images": [item["image"] for item in batch],
        "questions": [item["question"] for item in batch],
        "answers": [item.get("answers", []) for item in batch],
        "multiple_choice_answer": [
            item.get("multiple_choice_answer", None) for item in batch
        ],
        "choices": [item.get("choices", None) for item in batch],
    }
    return batch_dict


def sqa_collate_fn(batch):
    images, question_ids, questions, choices, answer_idxs, answers, contexts, lectures, solutions = \
        zip(*[
            (
                i["image"],
                i["question_id"],
                i["question"],
                i["choices"],
                i["answer_idx"],
                i["answer"],
                i.get("context", ""),
                i.get("lecture", ""),
                i.get("solution", ""),
            )
            for i in batch
        ])
    batch_dict = {
        "images": list(images),
        "question_ids": list(question_ids),
        "questions": list(questions),
        "choices": list(choices),
        "answer_idxs": list(answer_idxs),
        "answers": list(answers),
        "contexts": list(contexts),
        "lectures": list(lectures),
        "solutions": list(solutions),
    }
    return batch_dict


def mme_collate_fn(batch):
    """
    Collate function for MMEAllDataset (multi-task).
    """

    batch_dict = {
        "images": [b["image"] for b in batch],
        "questions": [b["question"] for b in batch],
        "answers": [b["answer"] for b in batch],
        "tasks": [b["task"] for b in batch],
        "ids": [b["id"] for b in batch],
    }

    task_indices = {}
    for i, t in enumerate(batch_dict["tasks"]):
        task_indices.setdefault(t, []).append(i)

    batch_dict["task_indices"] = task_indices
    return batch_dict


def mmbench_collate_fn(batch):
    """
    Collate function for MMBenchDataset
    - Supports image paths, questions, choices, hints, and comments
    - Keeps all textual metadata aligned with each sample
    """
    batch_dict = {
        "ids": [item["id"] for item in batch],
        "images": [item["image_path"] for item in batch],
        "questions": [item["question"] for item in batch],
        "choices": [item["choices"] for item in batch],  # list of ["A: xxx", "B: xxx", ...]
        "answers": [item["answer"] for item in batch],
        "hints": [item["hint"] for item in batch],
        "comments": [item["comment"] for item in batch],
        "tasks": [item["task"] for item in batch],
    }
    return batch_dict


def seedbench_collate_fn(batch):
    """
    Collate function for SEEDBenchDataset (image-only version).
    - Gathers image paths, questions, choices, answers, and question types.
    - Keeps all textual and metadata fields aligned per sample.
    """
    batch_dict = {
        "question_ids": [item["question_id"] for item in batch],
        "images": [item["image_path"] for item in batch],         # image paths only
        "questions": [item["question"] for item in batch],
        "choices": [item["choices"] for item in batch],           # list of ["A: ...", "B: ..."]
        "answers": [item["answer"] for item in batch],
        "question_types": [item["question_type"] for item in batch],  # e.g. "Instances Counting"
    }
    return batch_dict


def haloquest_collate_fn(batch):
    """
    Collate function for the HaloQuest dataset.
    - Gathers image paths or URLs, questions, groundtruth responses, hallucination types, etc.
    - Keeps all textual and metadata fields aligned per sample.
    """
    batch_dict = {
        "question_ids": [item["question_id"] for item in batch],
        "images": [item["image_path"] if item["image_path"] is not None else item["image_url"]
                   for item in batch],  # prefer local image if exists, else URL
        "questions": [item["question"] for item in batch],
        "answers": [item["groundtruth"] for item in batch],
        "hallucination_types": [item["hallucination_type"] for item in batch],
        "image_types": [item["image_type"] for item in batch],
        "splits": [item["split"] for item in batch],
    }
    return batch_dict


def collate_fn_mmhalbench(batch):
    """
    Collate function for MMHalBenchDataset.
    Simply stacks list-like fields while keeping strings and ints as lists.
    No tensorization (images are paths, not preprocessed yet).
    """
    images = [x["image_path"] for x in batch]
    questions = [x["question"] for x in batch]
    answers = [x["gt_answer"] for x in batch]
    question_types = [x["question_type"] for x in batch]
    question_topics = [x["question_topic"] for x in batch]
    image_contents = [x["image_content"] for x in batch]
    image_ids = [x["image_id"] for x in batch]

    return {
        "images": images,                # list[str]
        "questions": questions,          # list[str]
        "answers": answers,              # list[str]
        "question_types": question_types,  # list[str]
        "question_topics": question_topics,  # list[str]
        "image_contents": image_contents,  # list[str]
        "image_ids": image_ids,          # list[str]
    }
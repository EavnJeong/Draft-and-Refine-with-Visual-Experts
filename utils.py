import re 


def format_results(args, batch, answers):
    results = []
    
    if args.dataset in ['vqav2', 'okvqa']:
        for q_id, answer in zip(batch['question_ids'], answers):
            results.append({
                'question_id': int(q_id),
                'answer': answer.strip().lower(),
            })
    elif args.dataset == 'vizwiz':
        for image, answer in zip(batch['images'], answers):
            results.append({
                'image': image.split('/')[-1].strip(),
                'answer': answer.strip().lower(),
            })
    elif args.dataset == 'gqa':
        for qid, answer in zip(batch['question_id'], answers):
            results.append({
                'question_id': str(qid),
                'answer': answer.strip().lower()
            })
    elif args.dataset in ['textvqa', 'ocrvqa', 'vsr']:
        results.append({
            'predictions': answers,
            'answers': batch['answers'],
        })
    elif args.dataset in ['vcr']:
        results.append({
            'pred_answer': [int(ans.split('|')[0].split(':')[0][-1]) for ans in answers],
            'pred_rationale': [int(ans.split('|')[1].split(':')[0][-1]) for ans in answers],
            'answers': batch['answer_labels'],
            'rationale_labels': batch['rationale_labels'],
        })
    elif args.dataset == 'aokvqa':
        for q_id, answer in zip(batch['question_ids'], answers):
            results.append({
                'question_id': str(q_id),
                'answer': answer.strip().lower(),
            })
    elif args.dataset == 'sqa':
        results.append({
            'predictions': answers,
            'answers': batch['answers'],
        })
    elif args.dataset == 'mme':
        for task, sample_id, question, pred, gt in zip(
            batch["tasks"],
            batch["ids"],
            batch["questions"],
            answers,
            batch["answers"]
        ):
            results.append({
                "task": str(task).strip(),
                "id": str(sample_id).strip(),
                "question": question.strip(),
                "pred": pred.strip().lower(),
                "gt": gt.strip().lower(),
            })
    elif args.dataset == 'mmbench':
        for task, sample_id, question, choices, pred, gt in zip(
                batch.get("tasks", ["MMBench"] * len(answers)),
                batch["ids"],
                batch["questions"],
                batch.get("choices", [[]] * len(answers)),
                answers,
                batch["answers"],
            ):
                results.append({
                    "task": str(task).strip(),
                    "id": str(sample_id).strip(),
                    "question": question.strip(),
                    "choices": [c.strip() for c in choices] if choices else [],
                    # --- 🔧 extract only letter A/B/C/D if present ---
                    "pred": (
                        re.split(r"[:\s]", pred.strip())[0].upper()
                        if re.match(r"^[A-Da-d][:\s]?", pred.strip())
                        else pred.strip()
                    ),
                    "gt": gt.strip(),
                })        
    elif args.dataset == 'seedbench':
        for qid, qtype, pred, gt in zip(
            batch["question_ids"],
            batch["question_types"],
            answers,
            batch["answers"]
        ):
            results.append({
                "question_id": str(qid).strip(),
                "question_type": str(qtype).strip(),
                "pred": pred.strip().upper(),    # model prediction
                "gt": gt.strip().upper(),        # ground-truth answer
            })
    elif args.dataset == 'haloquest':
        for i in range(len(batch["questions"])):
            results.append({
                "question": batch["questions"][i].strip(),
                "answers": batch["answers"][i].strip(),
                "hallucination_type": batch["hallucination_types"][i],
                "pred": answers[i].strip().lower(),
            })
    elif args.dataset == "mmhalbench":
        for qtype, qtopic, question, pred, gt, imgpath in zip(
            batch["question_types"],     # e.g. "attribute", "counting", ...
            batch["question_topics"],    # e.g. "outdoor", "indoor", ...
            batch["questions"],          # question text
            answers,                     # model predictions
            batch["answers"],         # ground truth answers
            batch["images"],        # local image path
        ):
            results.append({
                "question_type": str(qtype).strip(),
                "question_topic": str(qtopic).strip(),
                "question": str(question).strip(),
                "model_answer": str(pred).strip(),
                "gt_answer": str(gt).strip(),
                "image_path": str(imgpath).strip(),
            })
    else:
        raise NotImplementedError

    return results
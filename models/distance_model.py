from sentence_transformers import SentenceTransformer
import open_clip


def get_dist_model(dist_model_name, device='cuda:0'):
    if dist_model_name == 'clip':
        model, _, _ = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained='openai'
        )
        model = model.to(device).eval()
        tokenizer = open_clip.get_tokenizer('ViT-L-14')
        return model, tokenizer

    elif dist_model_name == 'sbert':
        model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
        return model, None

    else:
        raise ValueError(f"Unknown dist_model_name: {dist_model_name}")
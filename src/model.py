from sentence_transformers import SentenceTransformer

def get_sentence_transformer(model_name_or_path, device=None):
    model = SentenceTransformer(model_name_or_path)
    if device is not None:
        model = model.to(device)
    return model
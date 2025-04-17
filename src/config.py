class Config:
    OUTPUT_PATH = "./finetuned_model_neg5"
    EPOCH = 5
    BS = 32
    TOP_K = 5
    GRAD_ACC_STEP = 6
    LR = 1e-4
    EARLY_STOPPING_PATIENCE = 2
    BASE_MODEL_NAME = "keepitreal/vietnamese-sbert"
    # GENERAL_ADAPTER_PATH = "/kaggle/input/general-adapter-viir/output_model"
    # ADAPTER_PATHS = {
    #     "geography": "/kaggle/input/geography-1neg-viir/output_model",
    #     "history": "/kaggle/input/history-1neg-viir/output_model",
    #     "legal": "/kaggle/input/legal-viir/output_model",
    #     "natural sciences": "/kaggle/input/medical-1neg-viir/output_model",
    #     "medical": "/kaggle/input/medical-1neg-viir/output_model",
    #     "miscellaneous": "/kaggle/input/general-adapter-viir/output_model"
    # }

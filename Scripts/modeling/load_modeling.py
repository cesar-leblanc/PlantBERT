import transformers
import torch

def load_tokenizer(args, task, fold=None):
    model_path = _model_path(args, task)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    return tokenizer

def load_model(args, task, fold=None, tokenizer=None, dataset=None):
    model_path = _model_path(args, task)
    if task == "fill-mask":
        model = transformers.AutoModelForMaskedLM.from_pretrained(model_path)
        model.resize_token_embeddings(len(tokenizer))
    elif task == "text classification":
        labels = dataset['train'].features['labels'].names
        id2label = {i: label for i, label in enumerate(labels)}
        label2id = {label: i for i, label in enumerate(labels)}
        config = transformers.AutoConfig.from_pretrained(model_path, label2id=label2id, id2label=id2label)
        model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path, config=config)
        return model, config
    elif task == "predict habitat":
        model = transformers.pipeline("text-classification", model=model_path, tokenizer=model_path, top_k=args.k_habitat)
    else:
        model = transformers.pipeline("fill-mask", model=model_path, tokenizer=model_path, top_k=10*args.k_species)
    return model

def load_optimizer(args, model):
    learning_rate = float(args.learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    return optimizer

def load_scheduler(args, train_dataloader, optimizer):
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = args.epochs * num_update_steps_per_epoch
    lr_scheduler = transformers.get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    return lr_scheduler

# Helper function for model paths for loading either tokenizer or model
def _model_path(args, task):
    if task == "fill-mask":
        model_checkpoint = f"bert-{args.model.split('-')[0]}-uncased"
    elif task == "text classification":
        if 'masking' in args.pipeline:
            model_checkpoint = f"plantbert_fill_mask_model_{args.model}_{args.method}_{args.batch_size}_{args.learning_rate}_{fold}"
        else:
            model_checkpoint = f"bert-{args.model}-uncased"
    else:
        if task == "predict habitat":
            model_checkpoint = args.model_habitat
        else:
            model_checkpoint = args.model_species
    return f"../Models/{model_checkpoint}"
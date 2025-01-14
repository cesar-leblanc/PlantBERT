import transformers
import torch

def load_tokenizer(args, task, fold=None):
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
    tokenizer = transformers.AutoTokenizer.from_pretrained(f'../Models/{model_checkpoint}/')
    return tokenizer

def load_model(args, task, fold=None, tokenizer=None, dataset=None):
    if task == "fill-mask":
        model_checkpoint = f"bert-{args.model.split('-')[0]}-uncased"
        model = transformers.AutoModelForMaskedLM.from_pretrained(f'../Models/{model_checkpoint}/')
        model.resize_token_embeddings(len(tokenizer))
        return model
    elif task == "text classification":
        if 'masking' in args.pipeline:
            model_checkpoint = f"plantbert_fill_mask_model_{args.model}_{args.method}_{args.batch_size}_{args.learning_rate}_{fold}"
        else:
            model_checkpoint = f"bert-{args.model}-uncased"
        labels = dataset['train'].features['labels'].names
        id2label = {i: label for i, label in enumerate(labels)}
        label2id = {label: i for i, label in enumerate(labels)}
        config = transformers.AutoConfig.from_pretrained(f'../Models/{model_checkpoint}/', label2id=label2id, id2label=id2label)
        model = transformers.AutoModelForSequenceClassification.from_pretrained(f'../Models/{model_checkpoint}/', config=config)
        return model, config
    else:
        if task == "predict habitat":
            model_checkpoint = args.model_habitat
            model = transformers.pipeline("text-classification", model=f"../Models/{model_checkpoint}", tokenizer=f"../Models/{model_checkpoint}", top_k=args.k_habitat)
            return model
        else:
            model_checkpoint = args.model_species
            model = transformers.pipeline("fill-mask", model=f"../Models/{model_checkpoint}", tokenizer=f"../Models/{model_checkpoint}", top_k=10*args.k_species)
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

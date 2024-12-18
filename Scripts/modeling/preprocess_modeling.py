import numpy as np

def add_vocabulary(args, dataset, tokenizer):
    dataset = dataset['train']['text']
    if args.model.split('-')[1] == "term":
        all_tokens = [item.replace(',', '').split(' ') for item in dataset]
    else:
        all_tokens = [item.split(', ') for item in dataset]
    all_tokens = np.concatenate([np.array(sublist) for sublist in all_tokens])
    all_tokens = np.unique(all_tokens)
    new_tokens = []
    for token in all_tokens:
        new_tokens.append(token)
    _ = tokenizer.add_tokens(new_tokens)
    return tokenizer, new_tokens

def prepare_everything(accelerator, model, optimizer, train_dataloader, eval_dataloader, lr_scheduler):
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    return model, optimizer, train_dataloader, eval_dataloader, lr_scheduler

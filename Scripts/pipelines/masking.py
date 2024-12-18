from data.load_data import load_dataset
from data.preprocess_data import create_specific_dataset, preprocess_dataset, create_dataloaders
from modeling.load_modeling import load_tokenizer, load_model, load_optimizer, load_scheduler
from modeling.preprocess_modeling import add_vocabulary, prepare_everything
from modeling.save_modeling import save_model, save_tokenizer
from epoch.utils_epoch import create_progress_bar
from epoch.train_epoch import train_one_epoch
from epoch.test_epoch import test_one_epoch

class FillMask:
    def create_dataset(self, args, fold):
        fill_mask_dataset = load_dataset(args, "fill-mask")
        text_classification_dataset = load_dataset(args, "text classification")
        fill_mask_dataset = create_specific_dataset(args, "fill-mask", fold, fill_mask_dataset, text_classification_dataset)
        return fill_mask_dataset
    
    def create_tokenizer(self, args, dataset, fold):
        tokenizer = load_tokenizer(args, "fill-mask", fold)
        tokenizer, vocabulary = add_vocabulary(args, dataset, tokenizer)
        return tokenizer, vocabulary
    
    def prepare_data(self, args, dataset, tokenizer, vocabulary, accelerator):
        fill_mask_dataset = preprocess_dataset("fill-mask", dataset, tokenizer, accelerator)
        train_dataloader, eval_dataloader = create_dataloaders(args, "fill-mask", fill_mask_dataset, tokenizer, accelerator, vocabulary)
        return train_dataloader, eval_dataloader
    
    def create_model(self, args, accelerator, tokenizer, train_dataloader, eval_dataloader, fold):
        model = load_model(args, "fill-mask", fold, tokenizer)
        optimizer = load_optimizer(args, model)
        lr_scheduler = load_scheduler(args, train_dataloader, optimizer)
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = prepare_everything(accelerator, model, optimizer, train_dataloader, eval_dataloader, lr_scheduler)
        return model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    
    def train_model(self, args, model, dataset, train_dataloader, eval_dataloader, accelerator, optimizer, lr_scheduler):
        progress_bar = create_progress_bar(args, train_dataloader)
        time_fold = 0
        for epoch in range(args.epochs):
            time_epoch = train_one_epoch(model, train_dataloader, accelerator, optimizer, lr_scheduler, progress_bar)
            perplexity = test_one_epoch(args, "fill-mask", epoch, eval_dataloader, model, dataset, accelerator, time_epoch)
            time_fold += time_epoch
        return perplexity, time_fold
    
    def save_all(self, args, accelerator, model, tokenizer, fold):
        save_model(args, 'fill-mask', accelerator, model, fold)
        save_tokenizer(args, 'fill-mask', accelerator, tokenizer, fold)

    def clear_memory(self, accelerator, model, optimizer, train_dataloader, eval_dataloader, scheduler):
        accelerator.free_memory()
        del model, optimizer, train_dataloader, eval_dataloader, scheduler

    def run(self, args, fold, accelerator):
        accelerator.print('\n' + '*'*20)
        accelerator.print(f'* Masking - Fold {fold} *')
        accelerator.print('*'*20 + '\n')
        fill_mask_dataset = self.create_dataset(args, fold)
        tokenizer, vocabulary = self.create_tokenizer(args, fill_mask_dataset, fold)
        train_dataloader, eval_dataloader = self.prepare_data(args, fill_mask_dataset, tokenizer, vocabulary, accelerator)
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = self.create_model(args, accelerator, tokenizer, train_dataloader, eval_dataloader, fold)
        perplexity, training_time = self.train_model(args, model, fill_mask_dataset, train_dataloader, eval_dataloader, accelerator, optimizer, lr_scheduler)
        self.save_all(args, accelerator, model, tokenizer, fold)
        self.clear_memory(accelerator, model, optimizer, train_dataloader, eval_dataloader, lr_scheduler)
        return perplexity, training_time
from data.load_data import load_dataset
from data.preprocess_data import create_specific_dataset, preprocess_dataset, create_dataloaders
from modeling.load_modeling import load_tokenizer, load_model, load_optimizer, load_scheduler
from modeling.utils_modeling import eval_model
from modeling.preprocess_modeling import prepare_everything
from modeling.save_modeling import save_model, save_tokenizer
from epoch.utils_epoch import create_progress_bar
from epoch.train_epoch import train_one_epoch
from epoch.test_epoch import test_one_epoch

class TextClassification:
    def create_dataset(self, args, fold):
        text_classification_dataset = load_dataset(args, "text classification")
        text_classification_dataset = create_specific_dataset(args, "text classification", fold, fill_mask_dataset=None, text_classification_dataset=text_classification_dataset)
        return text_classification_dataset
    
    def create_tokenizer(self, args, dataset, fold):
        tokenizer = load_tokenizer(args, "text classification", fold)
        return tokenizer
    
    def prepare_data(self, args, dataset, tokenizer, accelerator):
        text_classification_dataset = preprocess_dataset("text classification", dataset, tokenizer, accelerator)
        train_dataloader, eval_dataloader = create_dataloaders(args, "text classification", text_classification_dataset, tokenizer)
        return text_classification_dataset, train_dataloader, eval_dataloader
    
    def create_model(self, args, accelerator, train_dataloader, eval_dataloader, dataset, fold):
        model, config = load_model(args, "text classification", fold, None, dataset)
        optimizer = load_optimizer(args, model)
        lr_scheduler = load_scheduler(args, train_dataloader, optimizer)
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = prepare_everything(accelerator, model, optimizer, train_dataloader, eval_dataloader, lr_scheduler)
        return model, config, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    
    def train_model(self, args, model, dataset, train_dataloader, eval_dataloader, accelerator, optimizer, lr_scheduler):
        progress_bar = create_progress_bar(args, train_dataloader)
        time_fold = 0
        for epoch in range(args.epochs):
            time_epoch = train_one_epoch(model, train_dataloader, accelerator, optimizer, lr_scheduler, progress_bar)
            accuracy = test_one_epoch(args, "text classification", epoch, eval_dataloader, model, dataset, accelerator, time_epoch)
            time_fold += time_epoch
        return accuracy, time_fold

    def evaluate_model(self, eval_dataloader, model, accelerator, config):
        accuracy_top_3, accuracy_level_2, accuracy_level_1, precision_macro, recall_macro, f1_macro = eval_model(eval_dataloader, model, accelerator, config)
        return accuracy_top_3, accuracy_level_2, accuracy_level_1, precision_macro, recall_macro, f1_macro
    
    def save_all(self, args, accelerator, model, tokenizer, fold):
        save_model(args, 'text classification', accelerator, model, fold)
        save_tokenizer(args, 'text classification', accelerator, tokenizer, fold)
        
    def clear_memory(self, accelerator, model, config, optimizer, train_dataloader, eval_dataloader, scheduler):
        accelerator.free_memory()
        del model, config, optimizer, train_dataloader, eval_dataloader, scheduler

    def run(self, args, fold, accelerator):
        accelerator.print('\n' + '*'*27)
        accelerator.print(f'* Classification - Fold {fold} *')
        accelerator.print('*'*27 + '\n')
        text_classification_dataset = self.create_dataset(args, fold)
        tokenizer = self.create_tokenizer(args, text_classification_dataset, fold)
        text_classification_dataset, train_dataloader, eval_dataloader = self.prepare_data(args, text_classification_dataset, tokenizer, accelerator)
        model, config, optimizer, train_dataloader, eval_dataloader, lr_scheduler = self.create_model(args, accelerator, train_dataloader, eval_dataloader, text_classification_dataset, fold)
        accuracy, training_time = self.train_model(args, model, text_classification_dataset, train_dataloader, eval_dataloader, accelerator, optimizer, lr_scheduler)
        accuracy_top_3, accuracy_level_2, accuracy_level_1, precision_macro, recall_macro, f1_macro = self.evaluate_model(eval_dataloader, model, accelerator, config)
        self.save_all(args, accelerator, model, tokenizer, fold)
        self.clear_memory(accelerator, model, config, optimizer, train_dataloader, eval_dataloader, lr_scheduler)
        return accuracy, accuracy_top_3, accuracy_level_2, accuracy_level_1, precision_macro, recall_macro, f1_macro, training_time
import torch
import tqdm
import evaluate
import time

def compute_metric(logits, batch, accelerator, config, metric, k, level, level_dict):
    _, predictions = torch.topk(logits, k=k, dim=-1)
        
    if k == 1:
        predictions = [config.id2label[prediction.item()] for prediction in predictions]
    else:
        predictions = [[config.id2label[prediction.item()] for prediction in sublist] for sublist in predictions]

    labels = [config.id2label[label.item()] for label in batch["labels"]]

    if level != 3:
        if k == 1:
            predictions = [prediction[:-(3-level)] for prediction in predictions]
        else:
            predictions = [[prediction[:-(3-level)] for prediction in sublist] for sublist in predictions]
        labels = [label[:-(3-level)] for label in labels]
    
    labels = [level_dict[label] for label in labels]
    
    if k == 1:
        predictions = [level_dict[prediction] for prediction in predictions]
    else:
        predictions = [[level_dict[prediction] for prediction in sublist] for sublist in predictions]
        predictions = [label if label in sublist else -1 for sublist, label in zip(predictions, labels)]
    
    predictions = torch.Tensor(predictions).to("cuda")
    labels = torch.Tensor(labels).to("cuda")

    predictions, references = accelerator.gather_for_metrics((predictions, labels))
    metric.add_batch(predictions=predictions, references=references)

def eval_model(eval_dataloader, model, accelerator, config):
    model.eval()
    
    accuracy_top_3_micro = evaluate.load("metrics/accuracy.py")
    accuracy_level_2_micro = evaluate.load("metrics/accuracy.py")
    accuracy_level_1_micro = evaluate.load("metrics/accuracy.py")
    
    precision_macro = evaluate.load("metrics/precision.py")
    recall_macro = evaluate.load("metrics/recall.py")
    f1_macro = evaluate.load("metrics/f1.py")

    level_3_dict = sorted(set([label for label in config.label2id.keys()]))
    level_3_dict = {label: index for index, label in enumerate(level_3_dict)}
    
    level_2_dict = sorted(set([label[:-1] for label in config.label2id.keys()]))
    level_2_dict = {label: index for index, label in enumerate(level_2_dict)}
    
    level_1_dict = sorted(set([label[:-2] for label in config.label2id.keys()]))
    level_1_dict = {label: index for index, label in enumerate(level_1_dict)}
    
    for batch in tqdm.tqdm(eval_dataloader, position=0):
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        
        compute_metric(logits, batch, accelerator, config, accuracy_top_3_micro, 3, 3, level_3_dict)
        compute_metric(logits, batch, accelerator, config, accuracy_level_2_micro, 1, 2, level_2_dict)
        compute_metric(logits, batch, accelerator, config, accuracy_level_1_micro, 1, 1, level_1_dict)
        
        compute_metric(logits, batch, accelerator, config, precision_macro, 1, 3, level_3_dict)
        compute_metric(logits, batch, accelerator, config, recall_macro, 1, 3, level_3_dict)
        compute_metric(logits, batch, accelerator, config, f1_macro, 1, 3, level_3_dict)
    
    accelerator.wait_for_everyone()
    
    accuracy_top_3_micro = accuracy_top_3_micro.compute()['accuracy'] * 100
    accuracy_level_2_micro = accuracy_level_2_micro.compute()['accuracy'] * 100
    accuracy_level_1_micro = accuracy_level_1_micro.compute()['accuracy'] * 100
    
    precision_macro = precision_macro.compute(average='macro')['precision'] * 100
    recall_macro = recall_macro.compute(average='macro')['recall'] * 100
    f1_macro = f1_macro.compute(average='macro')['f1'] * 100
    
    return accuracy_top_3_micro, accuracy_level_2_micro, accuracy_level_1_micro, precision_macro, recall_macro, f1_macro

def make_predictions(args, vegetation_plots, model, tokenizer, task):
    predictions = []
    if task == 'predict habitat':
        start_time = time.time()
        for vegetation_plot in tqdm.tqdm(vegetation_plots["Observations"], desc="Identifying habitat types"):
            inputs = tokenizer(vegetation_plot, truncation=True, max_length=512, return_tensors='pt')
            vegetation_plot = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
            last_comma_index = vegetation_plot.rfind(',')
            if last_comma_index != -1:
                vegetation_plot = vegetation_plot[:last_comma_index]
            prediction = model(vegetation_plot)[0]
            predictions.append([pred['label'] for pred in prediction])
        elapsed_time = time.time() - start_time
    else:
        start_time = time.time()
        for vegetation_plot in tqdm.tqdm(vegetation_plots["Observations"], desc="Predicting missing species"):
            inputs = tokenizer(vegetation_plot, truncation=True, max_length=512-2*args.k_species+1, return_tensors='pt')
            vegetation_plot = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
            if len(inputs['input_ids'][0]) >= 512-2*args.k_species+1:
                last_comma_index = vegetation_plot.rfind(',')
                vegetation_plot = vegetation_plot[:last_comma_index]
            vegetation_plot = vegetation_plot.split(', ')
            best_predictions = []
            for _ in range(args.k_species):
                max_score = 0
                best_prediction = None
                best_position = None
                for i in range(len(vegetation_plot) + 1):
                    masked_vegetation_plot = ', '.join(vegetation_plot[:i] + ['[MASK]'] + vegetation_plot[i:])
                    j = 0
                    while True:
                        prediction = model(masked_vegetation_plot)[j]
                        species = prediction['token_str']
                        if species in vegetation_plot:
                            j += 1
                        else:
                            break
                    score = prediction['score']
                    position = i
                    if score > max_score:
                        max_score = score
                        best_prediction = species
                        best_position = position
                best_predictions.append(best_prediction)
                vegetation_plot.insert(best_position, best_prediction)
            predictions.append(best_predictions)
        elapsed_time = time.time() - start_time
    return predictions, elapsed_time
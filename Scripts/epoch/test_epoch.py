import torch
import math
import evaluate

def test_one_epoch(args, task, epoch, eval_dataloader, model, dataset, accelerator, training_time):
    model.eval()
    if task == "fill-mask":
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(accelerator.gather(loss.repeat(int(args.batch_size))))
        accelerator.wait_for_everyone()
        
        losses = torch.cat(losses)
        losses = losses[: len(dataset['test'])]
        try:
            perplexity = math.exp(torch.mean(losses))
        except OverflowError:
            perplexity = float("inf")
        accelerator.print(f"Epoch {epoch} - Perplexity: {perplexity:.4f}, Time: {training_time:.4f}s")
        return perplexity
    else:
        metric = evaluate.load("metrics/accuracy.py")
        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions, references = accelerator.gather_for_metrics((torch.argmax(logits, dim=-1), batch['labels']))  
            metric.add_batch(predictions=predictions, references=references)
        accelerator.wait_for_everyone()
        
        accuracy = metric.compute()['accuracy'] * 100
        accelerator.print(f"Epoch {epoch} - Accuracy: {accuracy:.4f}%, Time: {training_time:.4f}s")
        return accuracy
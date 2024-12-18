import time
import torch

def train_one_epoch(model, train_dataloader, accelerator, optimizer, lr_scheduler, progress_bar):
    model.train()
    start_time = time.time()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_tensor = torch.tensor(elapsed_time).to(accelerator.device)
    all_elapsed_times = accelerator.gather(elapsed_time_tensor)
    avg_elapsed_time = torch.mean(all_elapsed_times).item()
    accelerator.wait_for_everyone()
    return avg_elapsed_time
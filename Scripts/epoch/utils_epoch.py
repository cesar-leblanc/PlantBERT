import tqdm

def create_progress_bar(args, train_dataloader):
    num_train_epochs = args.epochs
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch
    progress_bar = tqdm.auto.tqdm(range(num_training_steps))
    return progress_bar
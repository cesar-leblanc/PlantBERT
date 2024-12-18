def save_model(args, task, accelerator, model, fold):
    if task == 'fill-mask':
        model_name = f"plantbert_fill_mask_model_{args.model}_{args.method}_{args.batch_size}_{args.learning_rate}_{fold}"
    else:
        model_name = f"plantbert_text_classification_model_{args.model}_{args.method}_{args.batch_size}_{args.learning_rate}_{fold}"
    model_path = f"../Models/{model_name}"
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(model_path, save_function=accelerator.save)
    accelerator.wait_for_everyone()
    
def save_tokenizer(args, task, accelerator, tokenizer, fold):
    if task == 'fill-mask':
        tokenizer_name = f"plantbert_fill_mask_model_{args.model}_{args.method}_{args.batch_size}_{args.learning_rate}_{fold}"
    else:
        tokenizer_name = f"plantbert_text_classification_model_{args.model}_{args.method}_{args.batch_size}_{args.learning_rate}_{fold}"
    tokenizer_path = f"../Models/{tokenizer_name}"
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        tokenizer.save_pretrained(tokenizer_path)
    accelerator.wait_for_everyone()

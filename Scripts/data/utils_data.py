import torch
import transformers

def tokenize_mask(examples, tokenizer):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

def tokenize_classification(example, tokenizer):    
    return tokenizer(example["text"], truncation=True)

def group_texts(examples, chunk_size=512):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // chunk_size) * chunk_size
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

class CustomDataCollatorForLanguageModeling(transformers.DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm_probability=0.15, replacement_words=None):
        super().__init__(tokenizer, mlm_probability)
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.replacement_words = replacement_words if replacement_words is not None else ['species']

    def torch_mask_tokens(self, inputs, special_tokens_mask = None):
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        special_indices = (inputs == self.tokenizer.convert_tokens_to_ids(',')) | \
                          (inputs == self.tokenizer.convert_tokens_to_ids('[CLS]')) | \
                          (inputs == self.tokenizer.convert_tokens_to_ids('[SEP]'))
        special_tokens_mask = (special_tokens_mask.bool() | special_indices) if special_tokens_mask is not None else special_indices

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.tensor([self.tokenizer.convert_tokens_to_ids(word) for word in self.replacement_words])
        random_words = random_words[torch.randint(len(random_words), labels.shape, dtype=torch.long)]
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels
    
def insert_random_mask(batch, data_collator):
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = data_collator(features)
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}

def add_column(vegetation_plots, type, predictions, scores=None, positions=None, completed_plots=None):
    predictions = [', '.join(pred) for pred in predictions]
    if type == "habitat":
        vegetation_plots['Habitat'] = predictions
    else:
        scores = [', '.join([str(round(score * 100, 2)) for score in score_list]) for score_list in scores]
        positions = [', '.join([str(pos) for pos in pos_list]) for pos_list in positions]
        vegetation_plots['Likely missing species'] = predictions
        vegetation_plots['Scores of likely missing species'] = scores
        vegetation_plots['Positions of likely missing species'] = positions
        vegetation_plots['Completed vegetation plots'] = completed_plots
    return vegetation_plots
import datasets
import torch
import transformers
import accelerate
import random
import pandas as pd
import requests
import tqdm
import numpy as np
import verde as vd

from data.utils_data import tokenize_mask, tokenize_classification, group_texts, CustomDataCollatorForLanguageModeling, insert_random_mask

def create_specific_dataset(args, task, fold, fill_mask_dataset=None, text_classification_dataset=None):
    if task == "fill-mask":
        text_classification_dataset = text_classification_dataset.remove_columns('label')
        fill_mask_dataset_train = [fill_mask_dataset['train']] + [text_classification_dataset[f'fold_{i}'] for i in range(0, len(text_classification_dataset)) if i != fold]
        fill_mask_dataset['train'] = datasets.concatenate_datasets(fill_mask_dataset_train)
        fill_mask_dataset['test'] = text_classification_dataset[f'fold_{fold}']
        if args.method == 'dropout':
            fill_mask_dataset['train'] = fill_mask_dataset['train'].map(lambda example: {'text': ', '.join([species for species in example['text'].split(', ') if random.random() > 0.3])})
            fill_mask_dataset['test'] = fill_mask_dataset['test'].map(lambda example: {'text': ', '.join([species for species in example['text'].split(', ') if random.random() > 0.3])})
        elif args.method == 'random':
            fill_mask_dataset['train'] = fill_mask_dataset['train'].map(lambda example: {'text': ', '.join(sorted(example['text'].split(', '), key=lambda x: random.random()))})
            fill_mask_dataset['test'] = fill_mask_dataset['test'].map(lambda example: {'text': ', '.join(sorted(example['text'].split(', '), key=lambda x: random.random()))})
        return fill_mask_dataset
    else:
        text_classification_dataset_train = [text_classification_dataset[f'fold_{i}'] for i in range(0, len(text_classification_dataset)) if i != fold]
        text_classification_dataset['train'] = datasets.concatenate_datasets(text_classification_dataset_train)
        text_classification_dataset['test'] = text_classification_dataset[f'fold_{fold}']
        text_classification_dataset = datasets.DatasetDict({key: text_classification_dataset[key] for key in ['train', 'test']})
        if args.method == 'dropout':
            text_classification_dataset['train'] = text_classification_dataset['train'].map(lambda example: {'text': ', '.join([species for species in example['text'].split(', ') if random.random() > 0.3])})
            text_classification_dataset['test'] = text_classification_dataset['test'].map(lambda example: {'text': ', '.join([species for species in example['text'].split(', ') if random.random() > 0.3])})
        elif args.method == 'random':
            text_classification_dataset['train'] = text_classification_dataset['train'].map(lambda example: {'text': ', '.join(sorted(example['text'].split(', '), key=lambda x: random.random()))})
            text_classification_dataset['test'] = text_classification_dataset['test'].map(lambda example: {'text': ', '.join(sorted(example['text'].split(', '), key=lambda x: random.random()))})
        return text_classification_dataset
    
def preprocess_dataset(task, dataset, tokenizer, accelerator):
    if task == 'fill-mask':
        with accelerator.main_process_first():
            tokenized_datasets = dataset.map(lambda examples: tokenize_mask(examples, tokenizer), batched=True, remove_columns=["text"])
            lm_datasets = tokenized_datasets.map(group_texts, batched=True)
        return lm_datasets
    else:
        with accelerator.main_process_first():
            tokenized_dataset = dataset.map(lambda example: tokenize_classification(example, tokenizer), batched=True)
        tokenized_dataset = tokenized_dataset.remove_columns(["text"])
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
        tokenized_dataset.set_format("torch")
        return tokenized_dataset
    
def create_dataloaders(args, task, dataset, tokenizer, accelerator=None, vocabulary=None):
    batch_size = int(args.batch_size)
    if task == 'fill-mask':
        data_collator = CustomDataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15, replacement_words=vocabulary)
        dataset = dataset.remove_columns(["word_ids"])
        with accelerator.main_process_first():
            eval_dataset = dataset["test"].map(lambda batch: insert_random_mask(batch, data_collator), batched=True, remove_columns=dataset["test"].column_names)
        eval_dataset = eval_dataset.rename_columns(
            {
                "masked_input_ids": "input_ids",
                "masked_attention_mask": "attention_mask",
                "masked_labels": "labels",
                "masked_token_type_ids": "token_type_ids"
            }
        )
        train_dataloader = torch.utils.data.DataLoader(dataset["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator)
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, collate_fn=transformers.default_data_collator)
    else:
        data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)
        train_dataloader = torch.utils.data.DataLoader(dataset["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator)
        eval_dataloader = torch.utils.data.DataLoader(dataset["test"], batch_size=batch_size, collate_fn=data_collator)
    return train_dataloader, eval_dataloader

def gbif_normalization(vegetation_plots):
    base = "https://api.gbif.org/v1"
    api = "species"
    function = "match"
    parameter = "name"
    url = f"{base}/{api}/{function}?{parameter}="
    if 'PlotObservationID' in vegetation_plots.columns:
        species_df = vegetation_plots['Species'].unique()
    else:
        species_df = pd.Series(", ".join(vegetation_plots["Observations"]).split(", ")).str.strip().unique()
    species_gbif = []
    for species in tqdm.tqdm(species_df, desc="GBIF normalization", position=0, leave=True):
        url = url.replace(url.partition('name')[2], f'={species}')
        r = requests.get(url)
        r = r.json()
        if 'species' in r:
            r = r["species"].lower()
        else:
            if 'PlotObservationID' in vegetation_plots.columns:
                r = '?'
            else:
                r = species
        species_gbif.append(r)
    df_to_gbif_species = dict(zip(species_df, species_gbif))
    if 'PlotObservationID' in vegetation_plots.columns:
        vegetation_plots['Species'] = vegetation_plots['Species'].map(df_to_gbif_species)
    else:
        vegetation_plots["Observations"] = vegetation_plots["Observations"].apply(lambda row: ", ".join(df_to_gbif_species.get(species, species) for species in row.split(", ")))
    return vegetation_plots

def keep_habitat_types(habitats):
    mask = ((habitats['Code'].str.startswith('MA2')) & (habitats['Code'].str.len() == 5)) | (~habitats['Code'].str.startswith('MA2') & (habitats['Code'].str.len() == 3))
    habitats = habitats[mask].reset_index(drop=True)
    return habitats

def split_header(header, habitats):
    correct_codes = habitats['Code'].values
    classification_header = header.loc[header['Habitat'].isin(correct_codes)].reset_index(drop=True)
    mask_header = header.loc[~header['Habitat'].isin(correct_codes)].reset_index(drop=True)
    mask_header = mask_header.drop(columns=['Habitat', 'Longitude', 'Latitude'])
    return mask_header, classification_header

def remove_absent_species(species):
    species = species[species["Cover"] > 0].reset_index(drop=True)
    return species

def aggregate_species(species):
    species = species.groupby(['PlotObservationID', 'Species']).sum().reset_index()
    return species

def snapshot_species(species):
    grouped_species = species.groupby('PlotObservationID')
    number_species = grouped_species.size()
    highest_cover = grouped_species.apply(lambda group: group.groupby('Species')['Cover'].sum().max())
    snapshot = pd.DataFrame({'NumberSpecies': number_species, 'HighestCover': highest_cover})
    snapshot = snapshot.reset_index()
    return snapshot

def remove_unknown_species(species):
    species = species[species['Species'] != '?'].reset_index(drop=True)
    return species

def remove_hybrid_species(species):
    species = species[species['Species'].str.split().str.len() == 2].reset_index(drop=True)
    return species

def remove_rare_species(args, species):
    counts = species['Species'].value_counts()
    keep = counts[counts >= args.occurrences].index.tolist()
    species = species[species['Species'].isin(keep)].reset_index(drop=True)
    return species

def compare_snapshots(snapshot_pre_normalization, snapshot_post_normalization, species):
    species = species.merge(snapshot_pre_normalization, on="PlotObservationID")
    species = species.merge(snapshot_post_normalization, on="PlotObservationID", suffixes=("PreNorm", "PostNorm"))
    species['PercentageChange'] = ((species['NumberSpeciesPreNorm'] - species['NumberSpeciesPostNorm']) / species['NumberSpeciesPreNorm']) * 100
    species = species[(species['PercentageChange'] <= 25) & (species['HighestCoverPreNorm'] == species['HighestCoverPostNorm'])]
    species = species[['PlotObservationID', 'Species', 'Cover']]
    species = species.reset_index(drop=True)
    return species

def remove_empty_plots(mask_header, classification_header, species):
    list_of_species_ids = species.PlotObservationID.unique()
    classification_header = classification_header[classification_header['PlotObservationID'].isin(list_of_species_ids)].reset_index(drop=True)
    mask_header = mask_header[mask_header['PlotObservationID'].isin(list_of_species_ids)].reset_index(drop=True)
    return mask_header, classification_header

def separate_rare_habitats(args, mask_header, classification_header):
    counts = classification_header['Habitat'].value_counts()
    keep = counts[counts >= args.occurrences].index.tolist()
    new_fill_mask_plot_ids = classification_header[~classification_header['Habitat'].isin(keep)].PlotObservationID.values
    new_fill_mask_rows = classification_header[classification_header['PlotObservationID'].isin(new_fill_mask_plot_ids)][['PlotObservationID']]
    classification_header = classification_header[~classification_header['PlotObservationID'].isin(new_fill_mask_plot_ids)].reset_index(drop=True)
    mask_header = pd.concat([mask_header, new_fill_mask_rows], ignore_index=True)
    return mask_header, classification_header

def sort_species(species):
    species = species.sort_values(by=['PlotObservationID', 'Cover'], ascending=[True, False])
    return species

def create_mask_df(mask_header, species):
    list_of_fill_mask_ids = mask_header.PlotObservationID.values
    mask_df = species[species['PlotObservationID'].isin(list_of_fill_mask_ids)]
    mask_df = mask_df.groupby(['PlotObservationID']).agg({'Species': ', '.join}).reset_index()
    mask_df = mask_df.rename(columns={'Species': 'text'})
    mask_df = pd.merge(mask_df, mask_header[['PlotObservationID']], on='PlotObservationID', how='left')
    mask_df = mask_df[['text']]
    return mask_df

def create_classification_df(classification_header, species):
    list_of_text_classification_ids = classification_header.PlotObservationID.values
    classification_species = species[species['PlotObservationID'].isin(list_of_text_classification_ids)]
    classification_df = classification_header.merge(classification_species)
    classification_df = classification_df.groupby(['PlotObservationID', 'Habitat', 'Longitude', 'Latitude']).agg({'Species': ', '.join}).reset_index()
    classification_df = classification_df.rename(columns={'Habitat': 'label', 'Species': 'text'})
    classification_df = classification_df[['label', 'text', 'Longitude', 'Latitude']]
    return classification_df

def perform_spatial_split(args, classification_df):
    coordinates = (classification_df.Longitude, classification_df.Latitude)
    kfold = vd.BlockKFold(spacing=args.spacing, n_splits=args.k_folds, shuffle=True, random_state=args.seed, balance=True)
    feature_matrix = np.transpose(coordinates)
    balanced = kfold.split(feature_matrix)
    splits = []
    for _, test in balanced:
        splits.append(test)
    split_dict = {index: split_value for split_value, array in enumerate(splits) for index in array}
    classification_df['split'] = classification_df.index.map(split_dict)
    classification_df = classification_df.drop(columns=['Longitude', 'Latitude'])
    return classification_df

def encode_target_labels(classification_df):
    num_labels = classification_df.label.nunique()
    list_labels = np.sort(classification_df.label.unique()).tolist()
    class_label_feature = datasets.ClassLabel(num_classes=num_labels, names=list_labels)
    return class_label_feature

def create_mask_dataset(mask_df):
    mask_dataset = datasets.DatasetDict()
    mask_dataset['train'] = datasets.Dataset.from_pandas(mask_df, preserve_index=False)
    return mask_dataset

def create_classification_dataset(args, classification_df, class_label_feature):
    classification_dataset = datasets.DatasetDict()
    for i in range(args.k_folds):
        classification_dataset[f"fold_{i}"] = datasets.Dataset.from_pandas(classification_df[classification_df['split'] == i][['label', 'text']], preserve_index=False)
        classification_dataset[f"fold_{i}"] = classification_dataset[f"fold_{i}"].cast_column("label", class_label_feature)
    return classification_dataset
import datasets
import pandas as pd

def load_header():
    header = pd.read_csv('../Data/header.csv')
    return header

def load_habitats():
    habitats = pd.read_excel('../Data/eunis_habitats.xlsx')
    return habitats

def load_species():
    species = pd.read_csv('../Data/species.csv')
    return species

def load_dataset(args, task):
    if task == "fill-mask":
        dataset_name = "plantbert_fill_mask_dataset"
    else:
        dataset_name = "plantbert_text_classification_dataset"
    dataset_path = f"../Datasets/{dataset_name}"
    dataset = datasets.load_from_disk(dataset_path)
    return dataset

def load_vegetation_plots():
    vegetation_plots = pd.read_csv("../Datasets/vegetation_plots.csv", sep="\t")
    return vegetation_plots
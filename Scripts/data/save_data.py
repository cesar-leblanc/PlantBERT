import pandas as pd
import datasets

def save_vegetation_plots(vegetation_plots):
    vegetation_plots.to_csv("../Datasets/vegetation_plots.csv", index=False, sep="\t")

def save_datasets(mask_dataset, classification_dataset):
    mask_dataset.save_to_disk("../Datasets/plantbert_fill_mask_dataset")
    classification_dataset.save_to_disk("../Datasets/plantbert_text_classification_dataset")
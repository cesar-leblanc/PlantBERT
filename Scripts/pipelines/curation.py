from utils import print_datasets_info

from data.load_data import load_header, load_habitats, load_species
from data.preprocess_data import keep_habitat_types, split_header, remove_absent_species, aggregate_species, snapshot_species, gbif_normalization, remove_unknown_species, remove_hybrid_species, remove_rare_species, compare_snapshots, remove_empty_plots, separate_rare_habitats, sort_species, create_mask_df, create_classification_df, perform_spatial_split, encode_target_labels, create_mask_dataset, create_classification_dataset
from data.save_data import save_datasets

class DataCuration:
    def load_data(self):
        header = load_header()
        habitats = load_habitats()
        species = load_species()
        return header, habitats, species

    def preprocess_habitats(self, habitats):
        habitats = keep_habitat_types(habitats)
        return habitats
    
    def preprocess_header(self, header, habitats):
        mask_header, classification_header = split_header(header, habitats)
        return mask_header, classification_header
    
    def preprocess_species(self, args, species):
        species = remove_absent_species(species)
        species = aggregate_species(species)
        snapshot_pre_normalization = snapshot_species(species)
        species = gbif_normalization(species)
        species = aggregate_species(species)
        species = remove_unknown_species(species)
        species = remove_hybrid_species(species)
        species = remove_rare_species(args, species)
        snapshot_post_normalization = snapshot_species(species)
        species = compare_snapshots(snapshot_pre_normalization, snapshot_post_normalization, species)
        return species
    
    def clean_data(self, args, mask_header, classification_header, species):
        mask_header, classification_header = remove_empty_plots(mask_header, classification_header, species)
        mask_header, classification_header = separate_rare_habitats(args, mask_header, classification_header)
        species = sort_species(species)
        return mask_header, classification_header, species
    
    def create_dataframes(self, args, mask_header, classification_header, species):
        mask_df = create_mask_df(mask_header, species)
        classification_df = create_classification_df(classification_header, species)
        classification_df = perform_spatial_split(args, classification_df)
        class_label_feature = encode_target_labels(classification_df)
        return mask_df, classification_df, class_label_feature
    
    def create_datasets(self, args, mask_df, classification_df, class_label_feature):
        mask_dataset = create_mask_dataset(mask_df)
        classification_dataset = create_classification_dataset(args, classification_df, class_label_feature)
        return mask_dataset, classification_dataset
    
    def datasets_info(self, accelerator, mask_dataset, classification_dataset):
        print_datasets_info(accelerator, mask_dataset, classification_dataset)
    
    def save_datasets(self, mask_dataset, classification_dataset):
        save_datasets(mask_dataset, classification_dataset)

    def run(self, args, accelerator):
        accelerator.print('\n' + '*'*17)
        accelerator.print(f'* Data curation *')
        accelerator.print('*'*17 + '\n')
        header, habitats, species = self.load_data()
        habitats = self.preprocess_habitats(habitats)
        mask_header, classification_header = self.preprocess_header(header, habitats)
        species = self.preprocess_species(args, species)
        mask_header, classification_header, species = self.clean_data(args, mask_header, classification_header, species)
        mask_df, classification_df, class_label_feature = self.create_dataframes(args, mask_header, classification_header, species)
        mask_dataset, classification_dataset = self.create_datasets(args, mask_df, classification_df, class_label_feature)
        self.datasets_info(accelerator, mask_dataset, classification_dataset)
        self.save_datasets(mask_dataset, classification_dataset)
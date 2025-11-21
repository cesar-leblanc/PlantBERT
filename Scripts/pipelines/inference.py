from data.load_data import load_vegetation_plots
from data.preprocess_data import gbif_normalization
from modeling.load_modeling import load_model, load_tokenizer
from modeling.utils_modeling import make_predictions
from data.utils_data import add_column
from data.save_data import save_vegetation_plots

class MakePredictions:
    def load_data(self):
        vegetation_plots = load_vegetation_plots()
        return vegetation_plots

    def clean_data(self, vegetation_plots):
        vegetation_plots = gbif_normalization(vegetation_plots)
        return vegetation_plots

    def predict_habitat(self, args, vegetation_plots):
        model = load_model(args, "predict habitat")
        tokenizer = load_tokenizer(args, "predict habitat")
        predictions, elapsed_time = make_predictions(args, vegetation_plots, model, tokenizer, "predict habitat")
        return predictions, elapsed_time

    def predict_species(self, args, vegetation_plots):
        model = load_model(args, "predict species")
        tokenizer = load_tokenizer(args, "predict species")
        predictions, scores, positions, completed_plots, elapsed_time =  make_predictions(args, vegetation_plots, model, tokenizer, "predict species")
        return predictions, scores, positions, completed_plots, elapsed_time

    def write_predictions(self, vegetation_plots, type, predictions, scores=None, positions=None, completed_plots=None):
        vegetation_plots = add_column(vegetation_plots, type, predictions, scores, positions, completed_plots)
        return vegetation_plots

    def save_data(self, vegetation_plots):
        save_vegetation_plots(vegetation_plots)

    def run(self, args, accelerator):
        accelerator.print('\n' + '*'*13)
        accelerator.print('* Inference *')
        accelerator.print('*'*13 + '\n')
        vegetation_plots = self.load_data()
        vegetation_plots = self.clean_data(vegetation_plots)
        if args.predict_habitat:
            predictions_habitat, time_habitat = self.predict_habitat(args, vegetation_plots)
            accelerator.print(f"Predicting habitats took {time_habitat:.2f} seconds")
            vegetation_plots = self.write_predictions(vegetation_plots, "habitat", predictions_habitat)
        if args.predict_species:
            predictions_species, scores_species, positions_species, completed_plots_species, time_species = self.predict_species(args, vegetation_plots)
            accelerator.print(f"Predicting species took {time_species:.2f} seconds")
            vegetation_plots = self.write_predictions(vegetation_plots, "species", predictions_species, scores_species, positions_species, completed_plots_species)
        self.save_data(vegetation_plots)
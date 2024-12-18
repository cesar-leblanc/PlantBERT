import argparse
import accelerate

from cli import add_all_parsers
from utils import disable_caching, set_seed, print_parameters, disable_warnings, control_verbosity, print_fold_results, print_final_results

from pipelines.check import InstallationCheck
from pipelines.curation import DataCuration
from pipelines.masking import FillMask
from pipelines.classification import TextClassification
from pipelines.inference import MakePredictions

def check(args, accelerator):
    InstallationCheck().run(args, accelerator)

def curation(args, accelerator):
    DataCuration().run(args, accelerator)

def masking(args, fold, results, accelerator):
    results.update({"Perplexity": [], "Masking time (s)": []})
    perplexity, training_time = FillMask().run(args, fold, accelerator)
    results["Perplexity"].append(perplexity)
    results["Masking time (s)"].append(training_time)
    return results
    
def classification(args, fold, results, accelerator):
    results.update({"Accuracy (%)": [], "Accuracy top-3 (%)": [], "Accuracy level 2 (%)": [], "Accuracy level 1 (%)": [], "Precision macro": [], "Recall macro": [], "F1 macro": [], "Classification time (s)": []})
    accuracy, accuracy_top_3, accuracy_level_2, accuracy_level_1, precision_macro, recall_macro, f1_macro, training_time = TextClassification().run(args, fold, accelerator)
    results["Accuracy (%)"].append(accuracy)
    results["Accuracy top-3 (%)"].append(accuracy_top_3)
    results["Accuracy level 2 (%)"].append(accuracy_level_2)
    results["Accuracy level 1 (%)"].append(accuracy_level_1)
    results["Precision macro"].append(precision_macro)
    results["Recall macro"].append(recall_macro)
    results["F1 macro"].append(f1_macro)
    results["Classification time (s)"].append(training_time)
    return results

def inference(args, accelerator):
    MakePredictions().run(args, accelerator)

def run(args, accelerator):
    disable_caching(args)
    set_seed(args)
    print_parameters(args, accelerator)
    disable_warnings(args)
    control_verbosity(args)
    if 'check' in args.pipeline:
        check(args, accelerator)
    if 'curation' in args.pipeline:
        curation(args, accelerator)
    if 'masking' in args.pipeline or 'classification' in args.pipeline:
        results = {}
        for fold in range(args.folds):
            if 'masking' in args.pipeline:
                results = masking(args, fold, results, accelerator)
            if 'classification' in args.pipeline:
                results = classification(args, fold, results, accelerator)
            print_fold_results(args, fold, results, accelerator)
        print_final_results(args, results, accelerator)
    if 'inference' in args.pipeline:
        inference(args, accelerator)

if __name__ == "__main__":
    accelerator = accelerate.Accelerator()
    parser = argparse.ArgumentParser()
    add_all_parsers(parser)
    args = parser.parse_args()
    run(args, accelerator)

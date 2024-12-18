from utils import bool_type

def add_all_parsers(parser):
    _add_global_parser(parser)
    _add_pipeline_parser(parser)
    pipelines = parser.parse_known_args()[0].pipeline
    if 'check' in pipelines:
        _add_check_parser(parser)
    if 'curation' in pipelines:
        _add_curation_parser(parser)
    if 'masking' in pipelines:
        _add_masking_parser(parser)
    if 'classification' in pipelines and 'masking' not in pipelines:
        _add_classification_parser(parser)
    if 'inference' in pipelines:
        _add_inference_parser(parser)

def _add_global_parser(parser):
    group_global = parser.add_argument_group('Global parameters')
    group_global.add_argument('--seed', type=int, default=123)
    group_global.add_argument('--display_parameters', type=bool_type, default=True)
    group_global.add_argument('--display_warnings', type=bool_type, default=False)
    group_global.add_argument('--use_cache', type=bool_type, default=False)
    group_global.add_argument('--verbose', type=bool_type, default=False)

def _add_pipeline_parser(parser):
    group_pipeline = parser.add_argument_group('Pipeline parameters')
    group_pipeline.add_argument('--pipeline', required=True, choices=['check', 'curation', 'masking', 'classification', 'inference'], nargs='+')

def _add_check_parser(parser):
    group_check = parser.add_argument_group('Check parameters')
    group_check.add_argument('--check_dependencies', type=bool_type, default=True)
    group_check.add_argument('--check_files', type=bool_type, default=True)
    group_check.add_argument('--check_environment', type=bool_type, default=True)
    
def _add_curation_parser(parser):
    group_curation = parser.add_argument_group('Curation parameters')
    group_curation.add_argument('--k_folds', type=int, default=10, choices=range(2, 101))
    group_curation.add_argument('--spacing', type=float, default=0.1)
    group_curation.add_argument('--occurrences', type=int, default=10)

def _add_masking_parser(parser):
    group_masking = parser.add_argument_group('Masking parameters')
    group_masking.add_argument('--model', default="base-species", choices=['base-term', 'base-species', 'large-term', 'large-species'])
    group_masking.add_argument('--batch_size', type=int, default=8)
    group_masking.add_argument('--learning_rate', type=float, default=3e-5)
    group_masking.add_argument('--folds', type=int, default=10)
    group_masking.add_argument('--epochs', type=int, default=5)
    group_masking.add_argument('--method', default="dominance", choices=['dominance', 'random', 'dropout'])

def _add_classification_parser(parser):
    group_classification = parser.add_argument_group('Classification parameters')
    group_classification.add_argument('--model', default="base", choices=['base', 'large'])
    group_classification.add_argument('--batch_size', type=int, default=8)
    group_classification.add_argument('--learning_rate', type=float, default=3e-5)
    group_classification.add_argument('--folds', type=int, default=10)
    group_classification.add_argument('--epochs', type=int, default=5)
    group_classification.add_argument('--method', default="dominance", choices=['dominance', 'random', 'dropout'])

def _add_inference_parser(parser):
    group_inference = parser.add_argument_group('Inference parameters')
    group_inference.add_argument('--predict_habitat', type=bool_type, default=True)
    group_inference.add_argument('--predict_species', type=bool_type, default=True)
    group_inference.add_argument('--model_habitat', default="plantbert_text_classification_model")
    group_inference.add_argument('--model_species', default="plantbert_fill_mask_model")
    group_inference.add_argument('--k_habitat', type=int, default=1)
    group_inference.add_argument('--k_species', type=int, default=1)
from typing import List
import os
import re
import hydra
from omegaconf import DictConfig


TEMPLATE_PATH = './embeddings_validation_template.yaml'
OUTPUT_CONFIG_PATH = './conf/embeddings_validation/embeddings_validation.yaml'
EXPERIMENT_FILES_PATH = './data'
FILL_ME_TOKEN = '<FILL_ME>'

DEFAULT_CONFIG_ITEM_TEMPLATE = """  {experiment_name}:
    enabled: true
    read_params: 
      file_name: ${{hydra:runtime.cwd}}/data/{experiment_file_name}
    target_options: {{}}
    """

def remove_suffix(s: str, suffix: str) -> str:
    if s.endswith(suffix):
        return s[:-len(suffix)]
    return s  


def get_experiment_name_from_filename(filename: str) -> str:
    exp_name = remove_suffix(filename, '.pickle')
    exp_name = remove_suffix(exp_name, '_embeddings')
    return exp_name


def get_experiment_file_names(experiment_files_path: str) -> List[str]:
    return [fname for fname in os.listdir(experiment_files_path) 
            if fname.endswith('.pickle')]


def get_config_lines(experiment_file_names: List[str], 
                     config_item_template: str
                     ) -> List[str]:
    config_lines_lst = []
    for exp_file_name in experiment_file_names:
        exp_name = get_experiment_name_from_filename(exp_file_name)
        config_line = config_item_template.format(
            experiment_name=exp_name, experiment_file_name=exp_file_name)
        config_lines_lst.append(config_line)
    return config_lines_lst


def fill_template(config_lines: List[str], template_path: str, 
                  output_file_path: str, fill_me_token: str) -> None:
    with open(template_path, 'r') as f:
        template = f.read()
    config_lines_str = '\n'.join(config_lines)
    config = template.replace(fill_me_token, config_lines_str)
    out_dir_name = os.path.dirname(output_file_path)
    if not os.path.exists(out_dir_name):
        os.makedirs(out_dir_name)
    with open(output_file_path, 'w') as f:
        f.write(config)

def get_identity_filter():
    def identity_filter_filenames(experiment_file_names: List[str]) -> List[str]:
        return experiment_file_names
    return identity_filter_filenames

def get_regular_expression_based_filtering(pattern: str):
    def convert_pattern_format(pattern: str) -> str:
        return pattern.replace("STRING_PATTERN", "(.+?)").replace("INT_PATTERN", "(\d+)")

    pattern = convert_pattern_format(pattern)
    regex = re.compile(pattern)

    def regular_expression_based_filtering(experiment_file_names: List[str]) -> List[str]:
        return [fname for fname in experiment_file_names if regex.match(fname)]
    return regular_expression_based_filtering




@hydra.main(version_base='1.2', config_path=None, config_name=None)
def main(conf: DictConfig):
    filter_filenames = hydra.utils.instantiate(conf.filter)
    experiment_file_names = get_experiment_file_names(conf.experiment_files_path)
    experiment_file_names = filter_filenames(experiment_file_names)
    config_lines = get_config_lines(experiment_file_names, DEFAULT_CONFIG_ITEM_TEMPLATE)
    fill_template(config_lines, conf.template_path, 
                  conf.output_config_path, conf.fill_me_token)


if __name__ == '__main__':
    main()

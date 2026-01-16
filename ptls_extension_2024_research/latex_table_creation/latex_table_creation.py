from typing import Dict, List, Optional, Callable
import os

from omegaconf import OmegaConf 


def create_latex_table(data: Dict[str, Dict[str, List[Dict[str, float]]]], 
                       hyperparameters: List[str], 
                       hyperparameter_header_strs: List[str],
                       caption: str,
                       row_prefix_dict: Optional[Dict[str, Dict[str, List[str]]]] = None) -> str:

    assert len(hyperparameters) ==  len(hyperparameter_header_strs)

    if row_prefix_dict is None:
        row_prefix_dict = {model: {weight_type: {} for weight_type in model_data} for model, model_data in data.items()}

    n_columns = 2 + len(hyperparameters)
    latex_table = \
        '\\begin{table*}[h!]' + '\n' + \
        '\\centering' + '\n' + \
        '\\resizebox{\\textwidth}{!}{%' + '\n' + \
        '\\begin{tabular}{|' + 'c|' * n_columns  + '}' + '\n' + \
        '\\hline' + '\n' \
        r'\textbf{Model} & \textbf{Type of weights} & ' + ' & '.join(hyperparameter_header_strs) + r'\\' + '\n' + \
        '\hline' + '\n'

    
    for model, model_data in data.items():
        model_started = False
        for weight_type, entries in model_data.items():
            weight_started = False
            for i, entry in enumerate(entries):
                # Format each row

                row = row_prefix_dict[model][weight_type].get(i, "")
                # row = ""

                if not model_started:
                    row += r"\multirow{" + str(len(entries)) + "}{*}{" + model + "} &"
                    model_started = True
                else:
                    row += "& "
                
                if not weight_started:
                    row += r"\multirow{" + str(len(entries)) + "}{*}{" + weight_type + "} & "
                    weight_started = True
                else:
                    row += "&"
                
                row += " & ".join([
                    str(entry[hyperparameter]) for hyperparameter in hyperparameters
                ]) 

                is_last_elemnt_of_current_weight_type = i == len(entries) - 1
                cline_start = 2 if is_last_elemnt_of_current_weight_type else 3
                cline_str = r" \\ \cline{" +str(cline_start) + f"-{n_columns}" + "} " 
                row += cline_str + "\n"
                
                latex_table += row
        
        latex_table += r"\hline" + "\n"
    
    
    latex_table += "\\end{tabular}" + "\n" + \
        "}" + "\n" + \
        "\\caption{" + caption + "}" + "\n" + \
        "\\label{tab:results}" + "\n" + \
        "\\end{table*}"
    
    return latex_table



def get_metrics(file_content: str, experiment_name: str, metric_report_name_to_metric_table_name: Dict[str, str]) -> dict:
    metrics = {}
    for metric_report_name, metric_table_name in metric_report_name_to_metric_table_name.items():
        metric_report_start = file_content.find(f"Metric: \"{metric_report_name}\"")
        scores_test_start  = file_content.find("scores_test", metric_report_start)
        target_line_start = file_content.find(experiment_name, scores_test_start)
        if target_line_start == -1:
            print(f"Could not find experiment {experiment_name} in file content")
            return None
        str_val = file_content[target_line_start + len(experiment_name):].split("\n")[0].split()[0]
        metrics[metric_table_name] = float(str_val)

    return metrics


def get_experiment_dicts_list(expected_experiments: List[str], 
                              hyperparams_to_getters: Dict[str, Callable],
                              hydra_outputs_path: str,
                              experiment_name_to_main_experiment_name: Callable,
                              report_file_path: str,
                              metric_report_name_to_metric_table_name: Dict[str, str]) -> List[Dict]:
    """
    Example of `experiment_name_to_main_experiment_name`:
    lambda exp_name: re.sub(r'(\d+)_epoches', '150_epoches', exp_name)
    """
    
    with open(report_file_path, 'r') as f:
        report_file_content = f.read()
    
    experiment_dicts_list = []

    for experiment_name in expected_experiments:
        experiment_name_main = experiment_name_to_main_experiment_name(experiment_name)
        hydra_config_path = os.path.join(hydra_outputs_path, experiment_name_main, '.hydra', 'config.yaml')
        if not os.path.exists(hydra_config_path):
            print(f"Warning! Config does not exists: {hydra_config_path}")
            config = None
        else: 
            config = OmegaConf.load(hydra_config_path)
        hyperparams = {k: v(config, experiment_name) for k, v in hyperparams_to_getters.items()}
        metrics = get_metrics(report_file_content, experiment_name, metric_report_name_to_metric_table_name)
        if metrics is not None:
            experiment_dicts_list.append({**hyperparams, **metrics})
    
    return experiment_dicts_list

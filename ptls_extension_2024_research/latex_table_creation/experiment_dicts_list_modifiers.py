from typing import List, Dict, Any


def assert_metric_in_data_and_number(data_lst: List[Dict[str, Any]], metric_name: str):
    for el in data_lst:
        assert metric_name in el, f"Metric name {metric_name} not found in data"
        assert isinstance(el[metric_name], (int, float)), f"Metric {metric_name} should be a number. Found {type(el[metric_name])}"


def sort_by_col(data_lst: List[Dict[str, Any]], metric_name: str):
    assert_metric_in_data_and_number(data_lst, metric_name)
    data_lst = sorted(data_lst, key=lambda x: x[metric_name], reverse=True)
    return data_lst


def bolden_top_k(data_lst: List[Dict[str, Any]], k, metric_names: List[str]):
    for metric_name in metric_names:
        assert_metric_in_data_and_number(data_lst, metric_name)

    for metric_name in metric_names:
        sorted_with_idxs = sorted(enumerate(data_lst), key=lambda x: x[1][metric_name], reverse=True)
        for i, data in sorted_with_idxs[:k]:
            data_lst[i][metric_name] = "\\textbf{" + str(data[metric_name]) + "}" 
    return data_lst
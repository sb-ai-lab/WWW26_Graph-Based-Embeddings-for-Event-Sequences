from typing import Dict


def get_idxs_where_all_metrics_superpass(data_lst, metric_name_to_limit_val: Dict[str, float]):
    idxs = []
    for i, data in enumerate(data_lst):
        superpass = True
        for metric_name, limit_val in metric_name_to_limit_val.items():
            if data[metric_name] < limit_val:
                superpass = False
                break
        if superpass:
            idxs.append(i)
    return idxs

def prefix_map_from_idx_lst(idx_lst, prefix):
    return {i: prefix for i in idx_lst}
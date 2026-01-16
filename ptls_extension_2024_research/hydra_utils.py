import pickle


def to_list(x):
    return list(x)

def load_pickle(f_path: str):
    with open(f_path, 'rb') as f:
        data = pickle.load(f)
    return data

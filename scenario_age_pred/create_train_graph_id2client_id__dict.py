import torch
import pandas as pd


train_trx_df = pd.read_parquet('data/train_trx_file.parquet')
train_client_ids = set(train_trx_df['encoded_client_id'])
client_id2train_graph_id = torch.load('data/graphs/weighted/client_id2train_graph_id.pt')
train_graph_id2client_id__dict = {int(client_id2train_graph_id[client_id]): client_id for client_id in train_client_ids}
torch.save(train_graph_id2client_id__dict, 'data/graphs/weighted/train_graph_id2client_id__dict.pt')

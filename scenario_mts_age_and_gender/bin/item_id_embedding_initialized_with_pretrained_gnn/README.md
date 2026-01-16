# Run all is legacy to be deleted, actual run_all is `run_al_freeze_and_unfreeze
run_all; run_all_freeze; run_all_unfreeze are for convinience on supercomputer


Possible scenarios:
1. gnn is pretrained is a separate pipeline from coles
2. coles+gnn pipeline is used for pretraining gnn 

In both scenarios after pretraining we initialize `url_host` embedding layer with pretrained gnn embeddings

Also during training we can have the embedding layer frozen for n epoches
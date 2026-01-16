import os
import argparse

import torch
import dgl


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--old_graph_path', type=str)
    parser.add_argument('--new_graph_path', type=str) 
    args = parser.parse_args(args)


def are_dgl_graphs_equal(g1, g2):
    if g1.to_networkx().is_directed() != g2.to_networkx().is_directed():
        return False
    
    if g1.num_nodes() != g2.num_nodes() or g1.num_edges() != g2.num_edges():
        return False
    
    if not torch.equal(g1.edges()[0], g2.edges()[0]) or not torch.equal(g1.edges()[1], g2.edges()[1]):
        return False
    
    if g1.ndata.keys() != g2.ndata.keys():
        return False
    for key in g1.ndata.keys():
        if not torch.equal(g1.ndata[key], g2.ndata[key]):
            return False
    
    if g1.edata.keys() != g2.edata.keys():
        return False
    for key in g1.edata.keys():
        if not torch.equal(g1.edata[key], g2.edata[key]):
            return False
    
    return True


if __name__ == "__main__":
    args = parse_args()

    for f_name in os.listdir(args.new_graph_path):
        new_path = os.path.join(args.new_graph_path, f_name)
        old_path = os.path.join(args.old_graph_path, f_name)

        if f_name.endswith('.pt'):
            are_equal = torch.allclose(torch.load(new_path), torch.load(old_path)) 
        elif f_name.endswith('.bin'):
            (new_g,), _ = dgl.load_graphs(new_path, [0])
            (old_g,), _ = dgl.load_graphs(old_path, [0])
            are_equal = are_dgl_graphs_equal(new_g, old_g)
        else:
            print(f"Unexpected extension {f_name}")
        
        if not are_equal:
            print(f"Files not equal, {f_name}")

            

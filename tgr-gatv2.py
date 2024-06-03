"""
Dynamic Link Prediction with a TGR-GATv2 model with Early Stopping
Reference: 
    - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py

command for an example run:
    python3 tgr-gatv2.py --data "tgbl-wiki" --num_run 1 --seed 1 --num_epoch=100 --lr=5e-4 
"""

from hmac import new
import math
import timeit
import wandb
import random
import argparse

import os
import os.path as osp
from pathlib import Path
import numpy as np
from tqdm import tqdm as tk

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import Linear
import torch.nn.functional as F

from torch_geometric.datasets import JODIEDataset
from torch_geometric.loader import TemporalDataLoader

from torch_geometric.nn import TransformerConv

# internal imports
from tgb.utils.utils import get_args, set_random_seed, save_results
from tgb.linkproppred.evaluate import Evaluator
import tqdm
from modules.decoder import LinkPredictor
from modules.emb_module import GraphAttentionEmbedding
from modules.msg_func import IdentityMessage
from modules.msg_agg import LastAggregator
from modules.neighbor_loader import LastNeighborLoader
from modules.memory_module import TGNMemory
from modules.early_stopping import  EarlyStopMonitor
from modules.shuffle_memory import ExpanderGCN, ExpanderGAT, ExpanderGIN, MLP, ExpanderGATv2
from modules.cayley_construction import build_cayley_bank, batched_augment_cayley
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset


# ==========
# ========== Define helper function...
# ==========

def train():
    r"""
    Training procedure for TGN model
    This function uses some objects that are globally defined in the current scrips 

    Parameters:
        None
    Returns:
        None
            
    """

    model['memory'].train()
    model['gnn'].train()
    model['link_pred'].train()

    model['memory'].reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    n_id_obs = torch.empty(0, dtype=torch.long, device=device) # Generate empty tensor to remember all observed nodes so far.
    z_exp_obs = torch.zeros(1, MEM_DIM, device=device) # Generate empty tensor to remember all expander embeddings so far.

    total_loss = 0
    for batch in tk(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        # Sample negative destination nodes.
        neg_dst = torch.randint(
            min_dst_idx,
            max_dst_idx + 1,
            (src.size(0),),
            dtype=torch.long,
            device=device,
        )

        n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc1[n_id_obs] = torch.arange(n_id_obs.size(0), device=device)
        new_nodes = n_id[~torch.isin(n_id, n_id_obs)] # Identify new nodes that have not been observed before.
        n_id_seen = n_id[~torch.isin(n_id, new_nodes)] # Find nodes in n-id which are not new nodes
        n_id_obs = torch.cat((n_id_obs, new_nodes.unique()), dim=0) # Append new nodes to the list of all nodes.
        
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Create input features z for TGNN forward pass
        z = torch.zeros(n_id.size(0), MEM_DIM, device=device)
        z_exp = z_exp_obs[assoc1[n_id_seen]].detach() # Get expander embeddings for nodes that have been observed before.
        z[assoc[new_nodes]] = model['memory'](new_nodes)[0] # Get node states for new nodes.
        z[assoc[n_id_seen]] = z_exp # Get node states for nodes that have been observed before.

        last_update = model['memory'](n_id)[1]
       
       #TGNN forward pass
        z = model['gnn'](
            z,
            last_update,
            edge_index,
            data.t[e_id].to(device),
            data.msg[e_id].to(device),
        )

        pos_out = model['link_pred'](z[assoc[src]], z[assoc[pos_dst]])
        neg_out = model['link_pred'](z[assoc[src]], z[assoc[neg_dst]])

        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))

        # Update memory and neighbor loader with ground-truth state.
        model['memory'].update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)
        loss.backward()
        optimizer.step()

        # Memory mixing to generate expander embeddings
        x_obs = model['memory'](n_id_obs)[0]
        # Calculate the padding size
        padding_size = cayley_g.max().item() + 1 - x_obs.shape[0]

        # Pad x_obs if necessary
        if padding_size > 0:
            x_obs = F.pad(x_obs, (0, 0, 0, padding_size))
        
        # Get expander embeddings for observed nodes.
        z_exp_obs = exp_gnn(x_obs, cayley_g) # Generate expander embeddings for observed nodes.

        model['memory'].detach()
        total_loss += float(loss) * batch.num_events
    
    return total_loss / train_data.num_events


@torch.no_grad()
def test(loader, neg_sampler, split_mode):
    r"""
    Evaluated the dynamic link prediction
    Evaluation happens as 'one vs. many', meaning that each positive edge is evaluated against many negative edges

    Parameters:
        loader: an object containing positive attributes of the positive edges of the evaluation set
        neg_sampler: an object that gives the negative edges corresponding to each positive edge
        split_mode: specifies whether it is the 'validation' or 'test' set to correctly load the negatives
    Returns:
        perf_metric: the result of the performance evaluation
    """
    model['memory'].eval()
    model['gnn'].eval()
    model['link_pred'].eval()

    perf_list = []

    n_id_obs = torch.empty(0, dtype=torch.long, device=device) # Generate empty tensor to remember all observed nodes so far.
    z_exp_obs = torch.zeros(1, MEM_DIM, device=device) # Generate empty tensor to remember all expander embeddings so far.

    for pos_batch in tk(loader):
        pos_src, pos_dst, pos_t, pos_msg = (
            pos_batch.src,
            pos_batch.dst,
            pos_batch.t,
            pos_batch.msg,
        )

        n_id_pos = torch.cat([pos_src, pos_dst]).unique()
        new_nodes = n_id_pos[~torch.isin(n_id_pos, n_id_obs)] # Identify new nodes that have not been observed before.
        n_id_seen = n_id_pos[~torch.isin(n_id_pos, new_nodes)] # Find nodes in n-id which are not new nodes
        n_id_obs = torch.cat((n_id_obs, new_nodes.unique()), dim=0) # Append new nodes to the list of all nodes.
        assoc1[n_id_obs] = torch.arange(n_id_obs.size(0), device=device)

        neg_batch_list = neg_sampler.query_batch(pos_src, pos_dst, pos_t, split_mode=split_mode)

        for idx, neg_batch in enumerate(neg_batch_list):
            src = torch.full((1 + len(neg_batch),), pos_src[idx], device=device)
            dst = torch.tensor(
                np.concatenate(
                    ([np.array([pos_dst.cpu().numpy()[idx]]), np.array(neg_batch)]),
                    axis=0,
                ),
                device=device,
            )

            n_id = torch.cat([src, dst]).unique()
            n_id, edge_index, e_id = neighbor_loader(n_id)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)

            # Get updated memory of all nodes involved in the computation.
            z = torch.zeros(n_id.size(0), MEM_DIM, device=device)
            z_exp = z_exp_obs[assoc1[n_id_seen]].detach() # Get expander embeddings for nodes that have been observed before.
            
            neg_nodes = n_id[~torch.isin(n_id, n_id_seen)] # Identify new nodes that have not been observed before.

            # Create input features z for TGNN forward pass.
            z[assoc[neg_nodes]] = model['memory'](neg_nodes)[0] # Get node states for new nodes.
            z[assoc[n_id_seen]] = z_exp # Get node states for nodes that have been observed before.

            last_update = model['memory'](n_id)[1]

            # TGNN forward pass.
            z = model['gnn'](
                z,
                last_update,
                edge_index,
                data.t[e_id].to(device),
                data.msg[e_id].to(device),
            )

            y_pred = model['link_pred'](z[assoc[src]], z[assoc[dst]])

            # compute MRR
            input_dict = {
                "y_pred_pos": np.array([y_pred[0, :].squeeze(dim=-1).cpu()]),
                "y_pred_neg": np.array(y_pred[1:, :].squeeze(dim=-1).cpu()),
                "eval_metric": [metric],
            }
            perf_list.append(evaluator.eval(input_dict)[metric])

        # Update memory and neighbor loader with ground-truth state.
        model['memory'].update_state(pos_src, pos_dst, pos_t, pos_msg)
        neighbor_loader.insert(pos_src, pos_dst)

        # Memory mixing to generate expander embeddings
        x_obs = model['memory'](n_id_obs)[0]
        # Calculate the padding size
        padding_size = cayley_g.max().item() + 1 - x_obs.shape[0]

        # Pad x_obs if necessary
        if padding_size > 0:
            x_obs = F.pad(x_obs, (0, 0, 0, padding_size))

        #Generate expander embeddings for observed nodes.
        z_exp_obs = exp_gnn(x_obs, cayley_g) # Generate expander embeddings for observed nodes.

    perf_metrics = float(torch.tensor(perf_list).mean())

    return perf_metrics

# ==========
# ==========
# ==========


# Start...
start_overall = timeit.default_timer()

# ========== set parameters...
args, _ = get_args()
print("INFO: Arguments:", args)

DATA = "tgbl-wiki"
LR = args.lr
BATCH_SIZE = args.bs
K_VALUE = args.k_value  
NUM_EPOCH = args.num_epoch
SEED = args.seed
MEM_DIM = args.mem_dim
TIME_DIM = args.time_dim
EMB_DIM = args.emb_dim
TOLERANCE = args.tolerance
PATIENCE = args.patience
NUM_RUNS = args.num_run
NUM_NEIGHBORS = 10

MODEL_NAME = 'GATv2'
# ==========

# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up W&B 
wandb.login()

run = wandb.init(
    # Set the project where this run will be logged
    project="wiki-gat-exp-mem",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": LR,
        "epochs": NUM_EPOCH,
    },
)

# data loading
dataset = PyGLinkPropPredDataset(name=DATA, root="datasets")
train_mask = dataset.train_mask
val_mask = dataset.val_mask
test_mask = dataset.test_mask
data = dataset.get_TemporalData()
data = data.to(device)
metric = dataset.eval_metric

train_data = data[train_mask]
val_data = data[val_mask]
test_data = data[test_mask]

train_loader = TemporalDataLoader(train_data, batch_size=BATCH_SIZE)
val_loader = TemporalDataLoader(val_data, batch_size=BATCH_SIZE)
test_loader = TemporalDataLoader(test_data, batch_size=BATCH_SIZE)

# Ensure to only sample actual destination nodes as negatives.
min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

# neighborhood sampler
neighbor_loader = LastNeighborLoader(data.num_nodes, size=NUM_NEIGHBORS, device=device)

# define the model end-to-end
memory = TGNMemory(
    data.num_nodes,
    data.msg.size(-1),
    MEM_DIM,
    TIME_DIM,
    message_module=IdentityMessage(data.msg.size(-1), MEM_DIM, TIME_DIM),
    aggregator_module=LastAggregator(),
).to(device)

gnn = GraphAttentionEmbedding(
    in_channels=MEM_DIM,
    out_channels=EMB_DIM,
    msg_dim=data.msg.size(-1),
    time_enc=memory.time_enc,
).to(device)

#Compute cayley bank
cayley_bank = build_cayley_bank()

# Find number of nodes appearing in the training dataset
num_cayley = train_data.num_nodes

# Initialise expander graph (Cayley graph) for memory mixing 
cayley_g, cayley_edge_attr = batched_augment_cayley(num_cayley, cayley_bank)
cayley_g = torch.LongTensor(cayley_g).to(device)  
cayley_edge_attr = torch.LongTensor(cayley_edge_attr).to(device)
cayley_edge_attr = cayley_edge_attr.float()

# Initialise expander GNN pass for memory mixing
exp_gnn = ExpanderGATv2(in_channels=MEM_DIM, out_channels=EMB_DIM).to(device)

link_pred = LinkPredictor(in_channels=EMB_DIM).to(device)

model = {'memory': memory,
         'gnn': gnn,
         'link_pred': link_pred}

optimizer = torch.optim.Adam(
    set(model['memory'].parameters()) | set(model['gnn'].parameters()) | set(model['link_pred'].parameters()),
    lr=LR,
)
criterion = torch.nn.BCEWithLogitsLoss()

# Helper vector to map global node indices to local ones.
assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)
assoc1 = torch.empty(data.num_nodes, dtype=torch.long, device=device)


print("==========================================================")
print(f"=================*** TGR-{MODEL_NAME}: LinkPropPred: {DATA} ***=============")
print("==========================================================")

evaluator = Evaluator(name=DATA)
neg_sampler = dataset.negative_sampler

# for saving the results...
results_path = f'{osp.dirname(osp.abspath(__file__))}/saved_results'
if not osp.exists(results_path):
    os.mkdir(results_path)
    print('INFO: Create directory {}'.format(results_path))
Path(results_path).mkdir(parents=True, exist_ok=True)
results_filename = f'{results_path}/{MODEL_NAME}_{DATA}_results.json'

for run_idx in range(NUM_RUNS):
    print('-------------------------------------------------------------------------------')
    print(f"INFO: >>>>> Run: {run_idx} <<<<<")
    start_run = timeit.default_timer()

    # set the seed for deterministic results...
    torch.manual_seed(run_idx + SEED)
    set_random_seed(run_idx + SEED)

    # define an early stopper
    save_model_dir = f'{osp.dirname(osp.abspath(__file__))}/saved_models/'
    save_model_id = f'{MODEL_NAME}_{DATA}_{SEED}_{run_idx}'
    early_stopper = EarlyStopMonitor(save_model_dir=save_model_dir, save_model_id=save_model_id, 
                                    tolerance=TOLERANCE, patience=PATIENCE)

    # ==================================================== Train, Validation & Test ====================================================
    # loading the validation negative samples
    dataset.load_val_ns()
    
    # loading the test negative samples
    dataset.load_test_ns()

    val_perf_list = []
    max_val_perf = 0.3
    max_test_perf = 0
    best_epoch = 0
    count = 0
    train_time_list = []
    val_time_list = []

    start_train_val = timeit.default_timer()
    for epoch in range(1, NUM_EPOCH + 1):
        # training
        start_epoch_train = timeit.default_timer()
        loss = train()
        print(
            f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Training elapsed Time (s): {timeit.default_timer() - start_epoch_train: .4f}"
        )

        train_time = timeit.default_timer() - start_epoch_train
        train_time_list.append(train_time)
        # validation
        start_val = timeit.default_timer()
        perf_metric_val = test(val_loader, neg_sampler, split_mode="val")
        print(f"\tValidation {metric}: {perf_metric_val: .4f}")
        print(f"\tValidation: Elapsed time (s): {timeit.default_timer() - start_val: .4f}")
        val_perf_list.append(perf_metric_val)
        val_time = timeit.default_timer() - start_val
        val_time_list.append(val_time)
        #Check whether to test (our early stopping criterion)
        if(perf_metric_val>max_val_perf):
            max_val_perf = perf_metric_val
            #start testing
            start_test = timeit.default_timer()
            perf_metric_test = test(test_loader, neg_sampler, split_mode="test")
            print(f"\tTest: {metric}: {perf_metric_test: .4f}")
            test_time = timeit.default_timer() - start_test
            print(f"\tTest: Elapsed Time (s): {test_time: .4f}")
            count = 0
            best_epoch = epoch
            max_test_perf = perf_metric_test
        else:   
            count += 1
            if count == 50:
                break 

        wandb.log({"val mrr": perf_metric_val, "loss": loss})

    train_val_time = timeit.default_timer() - start_train_val
    print(f"Train & Validation: Elapsed Time (s): {train_val_time: .4f}")

    print(f"Best epoch: {best_epoch}, Max Validation {metric}: {max_val_perf: .4f}, Test {metric}: {max_test_perf: .4f}")

    print(f"INFO: >>>>> Run: {run_idx}, elapsed time: {timeit.default_timer() - start_run: .4f} <<<<<")
    print('-------------------------------------------------------------------------------')

print(f"Overall Elapsed Time (s): {timeit.default_timer() - start_overall: .4f}")
print("==============================================================")

import math
import timeit
from tqdm import tqdm 

import os
import os.path as osp
from pathlib import Path
import numpy as np

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import Linear

from torch_sparse import SparseTensor

from torch_geometric.loader import TemporalDataLoader

from tgb.utils.utils import get_args, set_random_seed, save_results
from tgb.linkproppred.evaluate import Evaluator
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

from modules.NCNDecoder.NCNPred import NCNPredictor

def train():

    model['memory'].train()
    model['gnn'].train()
    model['link_pred'].train()

    model['memory'].reset_state()  
    neighbor_loader.reset_state()  

    n_id_obs = torch.empty(0, dtype=torch.long, device=device) 
    z_exp_obs = torch.zeros(1, MEM_DIM, device=device) 

    total_loss = 0

    for batch in tqdm(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        neg_dst = torch.randint(
            min_dst_idx,
            max_dst_idx + 1,
            (src.size(0),),
            dtype=torch.long,
            device=device,
        )

        n_id = torch.cat([src, pos_dst, neg_dst]).unique() 
        n_id, edge_index, e_id = find_neighbor(neighbor_loader, n_id, HOP_NUM) 
        new_nodes = n_id[~torch.isin(n_id, n_id_obs)] 
        n_id_seen = n_id[~torch.isin(n_id, new_nodes)] 
        n_id_obs = torch.cat((n_id_obs, new_nodes), dim=0).unique()
        
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        id_num = n_id.size(0)

        z, last_update = model['memory'](n_id)
        z_exp = z_exp_obs[n_id_seen].detach() 
     
        z[assoc[n_id_seen]] = z_exp 

        z = model['gnn'](
            z,
            last_update,
            edge_index,
            data.t[e_id].to(device),
            data.msg[e_id].to(device),
        )
        
        src_re = assoc[src]
        pos_re = assoc[pos_dst]
        neg_re = assoc[neg_dst]

        def generate_adj_1_hop():
            loop_edge = torch.arange(id_num, dtype=torch.int64, device=device)
            mask = ~ torch.isin(loop_edge, edge_index)
            loop_edge = loop_edge[mask]
            loop_edge = torch.stack([loop_edge,loop_edge])
            if edge_index.size(1) == 0:
                adj = SparseTensor.from_edge_index(loop_edge).to_device(device)
            else:
                adj = SparseTensor.from_edge_index(torch.cat((loop_edge, edge_index, torch.stack([edge_index[1], edge_index[0]])),dim=-1)).to_device(device)
            return adj
        
        def generate_adj_0_1_hop():
            loop_edge = torch.arange(id_num, dtype=torch.int64, device=device)
            loop_edge = torch.stack([loop_edge,loop_edge])
            if edge_index.size(1) == 0:
                adj = SparseTensor.from_edge_index(loop_edge).to_device(device)
            else:
                adj = SparseTensor.from_edge_index(torch.cat((loop_edge, edge_index, torch.stack([edge_index[1], edge_index[0]])),dim=-1)).to_device(device)
            return adj
        
        def generate_adj_0_1_2_hop(adj):
            adj = adj.matmul(adj)
            return adj

        if NCN_MODE == 0:
            adj_0_1 = generate_adj_0_1_hop()
            adj_1 = generate_adj_1_hop()
            adjs = (adj_0_1, adj_1)
        elif NCN_MODE == 1:
            adj_1 = generate_adj_1_hop()
            adjs = (adj_1)
        elif NCN_MODE == 2:
            adj_0_1 = generate_adj_0_1_hop()
            adj_1 = generate_adj_1_hop()
            adj_0_1_2 = generate_adj_0_1_2_hop(adj_1)
            adjs = (adj_0_1, adj_1, adj_0_1_2)
        else: 
            raise ValueError('Invalid NCN Mode! Mode must be 0, 1, or 2.')

        pos_out = model['link_pred'](z, adjs, torch.stack([src_re,pos_re]), NCN_MODE)
        neg_out = model['link_pred'](z, adjs, torch.stack([src_re,neg_re]), NCN_MODE)
        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))

        model['memory'].update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        loss.backward()
        optimizer.step()

        x_obs = model['memory'].memory

        z_exp_obs = exp_gnn(x_obs, cayley_g) 

        model['memory'].detach()
        total_loss += float(loss) * batch.num_events

    return total_loss / train_data.num_events


@torch.no_grad()
def test(loader, neg_sampler, split_mode):

    model['memory'].eval()
    model['gnn'].eval()
    model['link_pred'].eval()

    perf_list = []

    n_id_obs = torch.empty(0, dtype=torch.long, device=device) 
    z_exp_obs = torch.zeros(1, MEM_DIM, device=device) 

    for pos_batch in tqdm(loader):
        pos_src, pos_dst, pos_t, pos_msg = (
            pos_batch.src,
            pos_batch.dst,
            pos_batch.t,
            pos_batch.msg,
        )

        n_id_pos = torch.cat([pos_src, pos_dst]).unique()
        new_nodes = n_id_pos[~torch.isin(n_id_pos, n_id_obs)] 
        n_id_seen = n_id_pos[~torch.isin(n_id_pos, new_nodes)] 
        n_id_obs = torch.cat((n_id_obs, new_nodes), dim=0).unique() 

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
            n_id, edge_index, e_id = find_neighbor(neighbor_loader, n_id, HOP_NUM)
            n_id_seen_neg = n_id_seen[torch.isin(n_id_seen, n_id)]
            assoc[n_id] = torch.arange(n_id.size(0), device=device)
            
            id_num = n_id.size(0)

            z, last_update = model['memory'](n_id)
            z_exp = z_exp_obs[n_id_seen_neg].detach() 
            z[assoc[n_id_seen_neg]] = z_exp 

            z = model['gnn'](
                z,
                last_update,
                edge_index,
                data.t[e_id].to(device),
                data.msg[e_id].to(device),
            )

            def generate_adj_1_hop():
                loop_edge = torch.arange(id_num, dtype=torch.int64, device=device)
                mask = ~ torch.isin(loop_edge, edge_index)
                loop_edge = loop_edge[mask]
                loop_edge = torch.stack([loop_edge,loop_edge])
                if edge_index.size(1) == 0:
                    adj = SparseTensor.from_edge_index(loop_edge).to_device(device)
                else:
                    adj = SparseTensor.from_edge_index(torch.cat((loop_edge, edge_index, torch.stack([edge_index[1], edge_index[0]])),dim=-1)).to_device(device)
                return adj
            
            def generate_adj_0_1_hop():
                loop_edge = torch.arange(id_num, dtype=torch.int64, device=device)
                loop_edge = torch.stack([loop_edge,loop_edge])
                if edge_index.size(1) == 0:
                    adj = SparseTensor.from_edge_index(loop_edge).to_device(device)
                else:
                    adj = SparseTensor.from_edge_index(torch.cat((loop_edge, edge_index, torch.stack([edge_index[1], edge_index[0]])),dim=-1)).to_device(device)
                return adj
            
            def generate_adj_0_1_2_hop(adj):
                adj = adj.matmul(adj)
                return adj

            if NCN_MODE == 0:
                adj_0_1 = generate_adj_0_1_hop()
                adj_1 = generate_adj_1_hop()
                adjs = (adj_0_1, adj_1)
            elif NCN_MODE == 1:
                adj_1 = generate_adj_1_hop()
                adjs = (adj_1)
            elif NCN_MODE == 2:
                adj_0_1 = generate_adj_0_1_hop()
                adj_1 = generate_adj_1_hop()
                adj_0_1_2 = generate_adj_0_1_2_hop(adj_1)
                adjs = (adj_0_1, adj_1, adj_0_1_2)
            else:
                raise ValueError('Invalid NCN Mode! Mode must be 0, 1, or 2.')

            y_pred = model['link_pred'](z, adjs, torch.stack([assoc[src], assoc[dst]]), NCN_MODE)

            input_dict = {
                "y_pred_pos": np.array([y_pred[0, :].squeeze(dim=-1).cpu()]),
                "y_pred_neg": np.array(y_pred[1:, :].squeeze(dim=-1).cpu()),
                "eval_metric": [metric],
            }
            perf_list.append(evaluator.eval(input_dict)[metric])

        model['memory'].update_state(pos_src, pos_dst, pos_t, pos_msg)
        neighbor_loader.insert(pos_src, pos_dst)

        x_obs = model['memory'].memory

        z_exp_obs = exp_gnn(x_obs, cayley_g) 

    perf_metrics = float(torch.tensor(perf_list).mean())

    return perf_metrics

def find_neighbor(neighbor_loader:LastNeighborLoader, n_id, k=1):
    for i in range(k-1):
        n_id, _, _ = neighbor_loader(n_id)
    neighbor_info = neighbor_loader(n_id)
    return neighbor_info

start_overall = timeit.default_timer()

args, _ = get_args()
print("INFO: Arguments:", args)

DATA = "tgbl-wiki"
LR = 10e-5
BATCH_SIZE = args.bs
K_VALUE = 10  
NUM_EPOCH = args.num_epoch
SEED = args.seed
MEM_DIM = args.mem_dim
TIME_DIM = args.time_dim
EMB_DIM = args.emb_dim
TOLERANCE = args.tolerance
PATIENCE = args.patience
NUM_RUNS = args.num_run
NUM_NEIGHBORS = 10
HOP_NUM = 2
NCN_MODE = 2
PER_VAL_EPOCH = 1


MODEL_NAME = 'TGR-TNCN'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

neighbor_loader = LastNeighborLoader(data.num_nodes, size=NUM_NEIGHBORS, device=device)

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

cayley_bank = build_cayley_bank()

num_cayley = data.num_nodes
 
cayley_g, cayley_edge_attr = batched_augment_cayley(num_cayley, cayley_bank)
cayley_g = torch.LongTensor(cayley_g).to(device)  
cayley_edge_attr = torch.LongTensor(cayley_edge_attr).to(device)
cayley_edge_attr = cayley_edge_attr.float()

exp_gnn = ExpanderGAT(in_channels=MEM_DIM, out_channels=EMB_DIM).to(device)

link_pred = NCNPredictor(in_channels=EMB_DIM, hidden_channels=EMB_DIM, 
                         out_channels=1, NCN_mode=NCN_MODE).to(device)

model = {'memory': memory,
         'gnn': gnn,
         'link_pred': link_pred}

optimizer = torch.optim.Adam(
    set(model['memory'].parameters()) | set(model['gnn'].parameters()) | set(model['link_pred'].parameters()),
    lr=LR,
)
criterion = torch.nn.BCEWithLogitsLoss()

assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)

print("==========================================================")
print(f"=================*** {MODEL_NAME}: LinkPropPred: {DATA} ***=============")
print("==========================================================")

evaluator = Evaluator(name=DATA)
neg_sampler = dataset.negative_sampler

results_path = f'{osp.dirname(osp.abspath(__file__))}/saved_results'
if not osp.exists(results_path):
    os.mkdir(results_path)
    print('INFO: Create directory {}'.format(results_path))
Path(results_path).mkdir(parents=True, exist_ok=True)
results_filename = f'{results_path}/{MODEL_NAME}_{DATA}_{NCN_MODE}_results.json'

for run_idx in range(NUM_RUNS):
    print('-------------------------------------------------------------------------------')
    print(f"INFO: >>>>> Run: {run_idx} <<<<<")
    start_run = timeit.default_timer()

    torch.manual_seed(run_idx + SEED)
    set_random_seed(run_idx + SEED)

    save_model_dir = f'{osp.dirname(osp.abspath(__file__))}/saved_models/'
    save_model_id = f'{MODEL_NAME}_{DATA}_{SEED}_{run_idx}_NCN_{NCN_MODE}'
    early_stopper = EarlyStopMonitor(save_model_dir=save_model_dir, save_model_id=save_model_id, 
                                    tolerance=TOLERANCE, patience=PATIENCE)

    dataset.load_val_ns()
    
    dataset.load_test_ns()

    val_perf_list = []
    max_val_perf = 0.1
    max_test_perf = 0
    best_epoch = 0
    count = 0
    train_time_list = []
    val_time_list = []

    start_train_val = timeit.default_timer()
    for epoch in range(1, NUM_EPOCH + 1):
        start_epoch_train = timeit.default_timer()
        loss = train()
        print(
            f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Training elapsed Time (s): {timeit.default_timer() - start_epoch_train: .4f}"
        )

        train_time = timeit.default_timer() - start_epoch_train
        train_time_list.append(train_time)
        start_val = timeit.default_timer()
        perf_metric_val = test(val_loader, neg_sampler, split_mode="val")
        print(f"\tValidation {metric}: {perf_metric_val: .4f}")
        print(f"\tValidation: Elapsed time (s): {timeit.default_timer() - start_val: .4f}")
        val_perf_list.append(perf_metric_val)
        val_time = timeit.default_timer() - start_val
        val_time_list.append(val_time)
        if(perf_metric_val>max_val_perf):
            max_val_perf = perf_metric_val
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
            if count == 5:
                break 

    train_val_time = timeit.default_timer() - start_train_val
    print(f"Train & Validation: Elapsed Time (s): {train_val_time: .4f}")

    print(f"Best epoch: {best_epoch}, Max Validation {metric}: {max_val_perf: .4f}, Test {metric}: {max_test_perf: .4f}")

    print(f"INFO: >>>>> Run: {run_idx}, elapsed time: {timeit.default_timer() - start_run: .4f} <<<<<")
    print('-------------------------------------------------------------------------------')

print(f"Overall Elapsed Time (s): {timeit.default_timer() - start_overall: .4f}")
print("==============================================================")

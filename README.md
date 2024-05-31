# TGR
## Abstract 
Evolving relations in real-world networks are often modelled by temporal graphs. Graph rewiring techniques have been utilised on Graph Neural Networks (GNNs) to improve expressiveness and increase model performance. In this work, we propose Temporal Graph Rewiring (TGR), the first approach for graph rewiring on temporal graphs. TGR enables communication between temporally distant nodes in a continuous time dynamic graph by utilising expander graph propagation to construct a message passing highway for message passing between distant nodes. Expander graphs are suitable candidates for rewiring as they help overcome the oversquashing problem often observed in GNNs. On the public tgbl-wiki benchmark, we show that TGR improves the performance of a widely used TGN model by a significant margin.
<img width="300" alt="Screenshot 2024-05-31 145641" src="https://github.com/kpetrovicc/TGR/assets/122844200/185d2e6a-bdf3-47cd-8c6a-7a663592d24c">


## Installation
Please use installation guidelines on https://tgb.complexdatalab.com/ to install environment

## Running experiments
In this repo we provide scripts for testing TGR on 4 different baselines: GAT, GCN, GIN and GATv2
### Example
```console
conda activate tgbenv
cd TGR
python3 tgr-gat.py --data "tgbl-wiki" --num_run 1 --seed 1 --num_epoch=100 --lr=5e-4 
```

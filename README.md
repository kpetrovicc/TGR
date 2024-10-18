# TGR

## Abstract 
Evolving relations in real-world networks are often modelled by temporal graphs. Temporal Graph Neural Networks (TGNNs) emerged to model evolutionary behaviour of such graphs by leveraging the message passing primitive at the core of Graph Neural Networks (GNNs). It is well-known that GNNs are vulnerable to several issues directly related to the input graph topology, such as under-reaching and over-squashing---we argue that these issues can often get exacerbated in temporal graphs, particularly as the result of stale nodes and edges. While graph rewiring techniques have seen frequent usage in GNNs to make the graph topology more favourable for message passing, they have not seen any mainstream usage on TGNNs. In this work, we propose Temporal Graph Rewiring (TGR), the first approach for graph rewiring on temporal graphs, to the best of our knowledge. TGR constructs message passing highways between temporally distant nodes in a continuous-time dynamic graph by utilizing expander graph propagation, a prominent framework used for graph rewiring on static graphs which makes minimal assumptions on the underlying graph structure. On the challenging TGB benchmark, TGR achieves state-of-the-art results on tgbl-review, tgbl-coin, tgbl-comment and tgbl-flight datasets at the time of writing. For tgbl-review, TGR has 50.5\% improvement in MRR over the base TGN model and 22.2\% improvement over the base TNCN model. The significant improvement over base models demonstrates clear benefits of temporal graph rewiring.

![fig1_small](https://github.com/user-attachments/assets/c41d6eba-18f7-42b9-8a6b-845f96a96b5a)

## Installation
Please use installation guidelines on https://tgb.complexdatalab.com/ to install environment _and_ TGB repo

## Running experiments
In this repo we provide scripts for testing TGR on 2 different TGNN base models: TGN and TNCN
### Example (TGR-TGN with batch nodes)
```console
python3 tgr-tgn-b.py --seed 1 --num_epoch=50 --lr=10e-5 
```
### Example (TGR-TGN with batch + 1-hop nodes)
```console
python3 tgr-tgn-bn.py --seed 1 --num_epoch=50 --lr=10e-5 
```

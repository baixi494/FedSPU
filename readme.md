# FedSPU
This repository provides a reproducible implementation of FedSPU, a client-level federated unlearning method based on client-specific model information, first-order gradient prototypes, and mini-batch gradient second-moment prototypes.

## 1. Repository Structure
FedSPU
├── run_fed_unlearn.py   
├── client_proto.py      
├── sophia_f.py          
├── server_unlearn.py    
├── model.py             
└── utils.py             

## 2. Environment
The implementation is based on Python and PyTorch.
Recommended environment:
- Python 3.9 or later
- PyTorch
- torchvision
- NumPy
- CUDA-capable GPU


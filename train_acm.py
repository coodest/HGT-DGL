# import scipy.io
# import urllib.request

import warnings
warnings.filterwarnings("ignore")  # remove warnings

from lp_common import LPLoader, LPEval
import dgl

# import math
import numpy as np
from model import *

# data_url = 'https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/ACM.mat'
# data_file_path = '/tmp/ACM.mat'

# urllib.request.urlretrieve(data_url, data_file_path)
# data = scipy.io.loadmat(data_file_path)

dgl_dict, labels = LPLoader().load_data()

G = dgl.heterograph(dgl_dict)
print(G)

# pvc = data["PvsC"].tocsr()
# p_selected = pvc.tocoo()
# generate labels
# labels = pvc.indices
labels = torch.tensor(labels).long()

# generate train/val/test split
# pid = p_selected.row
pid = list(range(len(labels)))
shuffle = np.random.permutation(pid)
train_idx = torch.tensor(shuffle[0:800]).long()
val_idx = torch.tensor(shuffle[800:900]).long()
test_idx = torch.tensor(shuffle[900:]).long()

device = torch.device("cuda:0")
G.node_dict = {}
G.edge_dict = {}
for ntype in G.ntypes:
    G.node_dict[ntype] = len(G.node_dict)
for etype in G.etypes:
    G.edge_dict[etype] = len(G.edge_dict)
    G.edges[etype].data["id"] = (
        torch.ones(G.number_of_edges(etype), dtype=torch.long) * G.edge_dict[etype]
    )

# Random initialize input feature
G = G.to(device)
for ntype in G.ntypes:
    emb = nn.Parameter(
        torch.Tensor(G.number_of_nodes(ntype), 400), requires_grad=False
    ).to(device)
    nn.init.xavier_uniform_(emb)
    G.nodes[ntype].data["inp"] = emb


model = HGT(
    G,
    n_inp=400,
    n_hid=200,
    n_out=labels.max().item() + 1,
    n_layers=2,
    n_heads=4,
    use_norm=True,
).to(device)
optimizer = torch.optim.AdamW(model.parameters())
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, total_steps=1000, max_lr=1e-3, pct_start=0.05
)

best_val_acc = 0
best_test_acc = 0
train_step = 0
for epoch in range(1000):
    logits = model(G, "edge")
    # The loss is computed only for labeled nodes.
    loss = F.cross_entropy(logits[train_idx], labels[train_idx].to(device))

    pred = logits.argmax(1).cpu()
    train_acc = (pred[train_idx] == labels[train_idx]).float().mean()
    val_acc = (pred[val_idx] == labels[val_idx]).float().mean()
    test_acc = (pred[test_idx] == labels[test_idx]).float().mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_step += 1
    scheduler.step(train_step)

    LPEval.eval(pred[test_idx].numpy(), labels[test_idx].numpy())

    # if best_val_acc < val_acc:
    #     best_val_acc = val_acc
    #     best_test_acc = test_acc

    # if epoch % 5 == 0:
    #     print('LR: %.5f Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f)' % (
    #         optimizer.param_groups[0]['lr'],
    #         loss.item(),
    #         train_acc.item(),
    #         val_acc.item(),
    #         best_val_acc.item(),
    #         test_acc.item(),
    #         best_test_acc.item(),
    #     ))

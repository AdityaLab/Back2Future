# Multiple regions but we take features for same region, datapoint only
import pickle, os
import networkx as nx
import dgl
import numpy as np
import torch as th
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from covid_utils import features, stack_rev_history
from gnnrnn.gnn_model import GNNRNN, MainFNNNetwork
from tslearn.metrics import dtw, cdist_dtw, soft_dtw, cdist_soft_dtw
from optparse import OptionParser

label_idx = features.index("death_jhu_incidence")
pred_feats = features

pred_feats = [
    "death_jhu_incidence",
    "inIcuCurrently",
    "recovered",
    "hospitalizedIncrease",
    "covidnet",
]



parser = OptionParser()
parser.add_option("-s", "--startweek", dest="startweek", type="int", default=21) # Don't change this unless absolutely necessary
parser.add_option("-l", "--lastweek", dest="lastweek", type="int", default=53)
parser.add_option("-n", "--num", dest="num", type="string", default="noname")
parser.add_option("-p", "--pretrainepoch", dest="pretrain_epochs", type="int", default=500)
parser.add_option("-f", "--teacherforce", dest="teacherforce", type="float", default=0.5)
parser.add_option("-c", "--cuda", dest="cuda", type="string", default="yes")
parser.add_option("-m", "--maxlen", dest="maxlen", type="int", default=5)
(options, args) = parser.parse_args()

EPS = 1e-8
EXPT_NAME = options.num
C = 5
REV_LR = 1e-2
EPOCHS = options.pretrain_epochs
FORCE_PROB = options.teacherforce

if options.cuda[0]=="y":
    device = "cuda" if th.cuda.is_available() else "cpu"
else:
    device = "cpu"

start_week, end_week = options.startweek, options.lastweek

regions = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC',
            'FL', 'GA', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA',
            'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE',
            'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK',
            'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT',
            'VA', 'WA', 'WV', 'WI', 'WY']

regions = ['CA', 'DC', 'FL', 'GA', 'IL', 'NY', 'TX', 'WA']

pred_feats = np.array([features.index(f) for f in pred_feats])
pred_feats = np.concatenate([pred_feats+i*pred_feats.shape[0] for i in range(len(regions))])

# get data till current week
print("Retrieve data....")
data = [
    np.array([stack_rev_history(w, r) for r in regions])[:,:(end_week-w+1)]
    for w in tqdm(range(start_week, end_week+1))
]
print("Preprocess data...")
# normalize features
means = np.mean(np.concatenate(data, 1), 1)
std = np.std(np.concatenate(data, 1), 1)
data = [(d - means[:, None, :]) / (std[:, None, :] + EPS) for d in data]

MAXLEN = options.maxlen
data = data[:-(MAXLEN-1)]

# get the graph
print("Extract graph...")
num_feats = data[0].shape[0]*data[0].shape[2]
data1 = [d.transpose(0,2,1).reshape(num_feats, -1) for d in data]
if os.path.exists(f"saves/{EXPT_NAME}_dtw_weights.pkl"):
    with open(f"saves/{EXPT_NAME}_dtw_weights.pkl", "rb") as fl:
        weights = pickle.load(fl)
else:
    weights = np.zeros((num_feats, num_feats))
    for d in tqdm(data1):
        weights+= cdist_dtw(d, n_jobs=-1)
    max_wt = weights.max()
    weights = weights + (np.tri(*weights.shape)*max_wt*100)


    with open(f"saves/{EXPT_NAME}_dtw_weights.pkl", "wb") as fl:
        pickle.dump(weights, fl)

num_edges = C*num_feats
max_edge_wt = np.sort(weights.ravel())[num_edges]
edges = (weights<max_edge_wt).astype(int)
graph = nx.convert_matrix.from_numpy_array(edges)

def save_rev_model(rev_model: GNNRNN, file_prefix: str = f"saves/{EXPT_NAME}"):
    th.save(rev_model.state_dict(), file_prefix + "_rev_model.pth")


def load_rev_model(rev_model:GNNRNN, file_prefix: str = f"saves/{EXPT_NAME}"):
    rev_model.load_state_dict(th.load(file_prefix + "_rev_model.pth"))

print("Start training...")
rev_model = GNNRNN(num_feats, 50, device=device).to(device)
rev_opt = optim.Adam(rev_model.parameters(), lr=REV_LR)

graph = dgl.from_networkx(graph).to(device)
th_data = np.zeros((len(data1), data1[0].shape[0], data1[0].shape[1]))
th_mask = np.zeros((len(data1), data1[0].shape[0], data1[0].shape[1]))
for i, d1 in enumerate(data1):
    th_data[i, :, :d1.shape[1]] = d1
    th_mask[i, :, :d1.shape[1]] = 1.
th_data = th.FloatTensor(th_data).to(device)
th_mask = th.FloatTensor(th_mask).to(device)

best_loss= 100.
for e in range(EPOCHS):
    loss = 0
    rev_opt.zero_grad()
    rev_preds, rev_hiddens = rev_model.forward(graph, th_data.transpose(1,2), teach=True)
    #error = F.mse_loss(rev_preds[:,:-1].transpose(1,2)*th_mask[:,:,1:], th_data[:,:,1:]*th_mask[:,:,1:])
    error = F.mse_loss(rev_preds[:,:-1, pred_feats].transpose(1,2)*th_mask[:,pred_feats,1:], th_data[:,pred_feats,1:]*th_mask[:,pred_feats,1:])
    error.backward()
    loss += error.cpu().detach().numpy()
    rev_opt.step()
    if loss < best_loss:
        save_rev_model(rev_model)
        best_loss = loss
    print(f"Epoch {e+1}/{EPOCHS} Loss: {loss}")
    if loss > best_loss + 0.1:
        break

load_rev_model(rev_model)

for e in range(EPOCHS):
    loss = 0
    rev_opt.zero_grad()
    rev_preds, rev_hiddens = rev_model.forward(graph, th_data.transpose(1,2), teach=False, prob=FORCE_PROB)
    #error = F.mse_loss(rev_preds[:,:-1].transpose(1,2)*th_mask[:,:,1:], th_data[:,:,1:]*th_mask[:,:,1:])
    error = F.mse_loss(rev_preds[:,:-1, pred_feats].transpose(1,2)*th_mask[:,pred_feats,1:], th_data[:,pred_feats,1:]*th_mask[:,pred_feats,1:])
    error.backward()
    loss += error.cpu().detach().numpy()
    if loss < best_loss:
        save_rev_model(rev_model)
        best_loss = loss
    print(f"Epoch {e+1}/{EPOCHS} Loss: {loss}")
    if loss > best_loss + 1.0:
        break



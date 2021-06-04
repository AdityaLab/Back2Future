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
from gnnrnn.gnn_model import GNNRNN, MainFNNNetwork, ModelBiasEncoder, Refiner
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
pred_feats = np.array([features.index(f) for f in pred_feats])

parser = OptionParser()
parser.add_option("-s", "--startweek", dest="startweek", type="int", default=21) # Don't change this unless absolutely necessary
parser.add_option("-l", "--lastweek", dest="lastweek", type="int", default=53)
parser.add_option("-n", "--num", dest="num", type="string", default="noname")
parser.add_option("-e", "--epochs", dest="epochs", type="int", default=500)
parser.add_option("-a", "--ahead", dest="ahead", type="int", default=2)
parser.add_option("-c", "--cuda", dest="cuda", type="string", default="yes")
parser.add_option("-m", "--maxlen", dest="maxlen", type="int", default=5)
parser.add_option("-p", "--predpath", dest="predpath", type="string", default="./gtdc_preds_2.pkl")
(options, args) = parser.parse_args()

EPS = 1e-8
EXPT_NAME = options.num
C = 5
L = 5
REV_LR = 1e-2
EPOCHS = options.epochs
AHEAD = options.ahead
PREDPATH = options.predpath
LR = 1e-3

if options.cuda[0]=="y":
    device = "cuda" if th.cuda.is_available() else "cpu"
else:
    device = "cpu"

start_week, end_week = options.startweek, options.lastweek

regions_all = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC',
            'FL', 'GA', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA',
            'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE',
            'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK',
            'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT',
            'VA', 'WA', 'WV', 'WI', 'WY']
regions = regions_all
regions = ['CA', 'DC', 'FL', 'GA', 'IL', 'NY', 'TX', 'WA']


# get data till current week
print("Retrieve data....")
data = [
    np.array([stack_rev_history(w, r) for r in regions])[:,:(end_week-w+1)]
    for w in tqdm(range(start_week, end_week+1))
]
gt = np.array([d[:,-1,label_idx] for d in data]).transpose(1,0)
real_gt = np.array([d[:,0,label_idx] for d in data]).transpose(1,0)
with open(PREDPATH, "rb") as fl:
    model_preds = pickle.load(fl)
model_preds_ahead = model_preds[[regions_all.index(r) for r in regions], (end_week-start_week+1+AHEAD)]
model_preds = model_preds[[regions_all.index(r) for r in regions], :(end_week-start_week+1)]

print("Preprocess data...")
# normalize features
means = np.mean(np.concatenate(data, 1), 1)
std = np.std(np.concatenate(data, 1), 1)
data = [(d - means[:, None, :]) / (std[:, None, :] + EPS) for d in data]
gt_norm, real_gt_norm, model_preds_norm = [(x - means[:,label_idx,None])/(std[:, label_idx,None]+EPS) for x in [gt, real_gt, model_preds]]
pred_seq = np.stack([gt_norm, real_gt_norm, model_preds_norm]).transpose(1,2,0)

# get the graph
print("Extract graph...")
num_feats = data[0].shape[0]*data[0].shape[2]
data1 = [d.transpose(0,2,1).reshape(num_feats, -1) for d in data]

with open(f"saves/{EXPT_NAME}_dtw_weights.pkl", "rb") as fl:
    weights = pickle.load(fl)

num_edges = C*num_feats
max_edge_wt = np.sort(weights.ravel())[num_edges]
edges = (weights<max_edge_wt).astype(int)
graph = nx.convert_matrix.from_numpy_array(edges)

def save_rev_model(rev_model: GNNRNN, file_prefix: str = f"saves/{EXPT_NAME}"):
    th.save(rev_model.state_dict(), file_prefix + "_rev_model.pth")


def load_rev_model(rev_model:GNNRNN, file_prefix: str = f"saves/{EXPT_NAME}"):
    rev_model.load_state_dict(th.load(file_prefix + "_rev_model.pth"))

def load_model(rev_model:GNNRNN, bias_encoder: ModelBiasEncoder, refiner: Refiner, file_prefix:  str = f"saves/{EXPT_NAME}"):
    rev_model.load_state_dict(th.load(file_prefix + "_fine_rev_model.pth"))
    bias_encoder.load_state_dict(th.load(file_prefix + "_fine_bias_encoder.pth"))
    refiner.load_state_dict(th.load(file_prefix + "_fine_refiner.pth"))

def save_model(rev_model:GNNRNN, bias_encoder: ModelBiasEncoder, refiner: Refiner, file_prefix:  str = f"saves/{EXPT_NAME}"):
    th.save(rev_model.state_dict(),file_prefix + "_fine_rev_model.pth")
    th.save(bias_encoder.state_dict(),file_prefix + "_fine_bias_encoder.pth")
    th.save(refiner.state_dict(),file_prefix + "_fine_refiner.pth")

print("Start training...")
graph = dgl.from_networkx(graph).to(device)
rev_model = GNNRNN(num_feats, 50, device=device).to(device)
bias_encoder = ModelBiasEncoder(50).to(device)
refiner = Refiner(num_feats, 50, 50, device=device).to(device)
all_opt = optim.Adam(list(rev_model.parameters()) + list(bias_encoder.parameters())+list(refiner.parameters()), lr=LR)
load_rev_model(rev_model)

best_loss = 1000.
for e in range(EPOCHS):
    all_opt.zero_grad()
    loss = 0.
    for w in tqdm(range(AHEAD+1, end_week-start_week+1)):
        # past predictions
        ps = pred_seq[:,:(w-AHEAD),:]
        bseq = data1[w-AHEAD]
        curr_pred, gt_week, real_week = model_preds_norm[:, w], gt_norm[:, w], real_gt_norm[:, w]
        rev_preds, rev_hiddens = rev_model.forward(graph, th.FloatTensor(bseq.T[None,:,:]).to(device), teach=True)
        bias_enc = bias_encoder.forward(th.FloatTensor(ps.transpose(1,0,2)).to(device))[-1]
        gamma = refiner.forward(bias_enc, rev_hiddens, th.FloatTensor(curr_pred).to(device))
        error = F.mse_loss((gamma+1.)*th.FloatTensor(curr_pred).to(device), th.FloatTensor(gt_week).to(device))
        error.backward()
        loss += error.cpu().detach().numpy()
    all_opt.step()
    print(f"Epoch {e+1}/{EPOCHS} Loss: {loss}")
    print(gamma)
    print(curr_pred)
    print(gt_week)
    if loss < best_loss:
        save_model(rev_model, bias_encoder, refiner)
        best_loss = loss

# Inference
with th.no_grad():
    ps = pred_seq[:,:-1,:]
    bseq = np.concatenate([data1[-1]]*L, 1)
    curr_pred = (model_preds_ahead-means[:,label_idx])/(std[:, label_idx]+EPS)
    rev_preds, rev_hiddens = rev_model.forward(graph, th.FloatTensor(bseq.T[None,:,:]).to(device), teach=False, prob=1.0)
    bias_enc = bias_encoder.forward(th.FloatTensor(ps.transpose(1,0,2)).to(device))[-1]
    gamma = refiner.forward(bias_enc, rev_hiddens, th.FloatTensor(curr_pred).to(device))
    gamma = gamma.cpu().detach().numpy()
    ref_pred = (1.+gamma)*curr_pred

curr_pred, ref_pred = [(x*std[:, label_idx]) + means[:, label_idx] for x in [curr_pred, ref_pred]]

print(f"Model Pred: {curr_pred}")
print(f"Refined Pred: {ref_pred}")

with open(f"results/{EXPT_NAME}_pred.pkl", "wb") as fl:
    pickle.dump({
        "Expt name": EXPT_NAME,
        "Weeks ahead": AHEAD,
        "regions": regions,
        "current week": end_week,
        "forecast": curr_pred,
        "refined": ref_pred
    }, fl)


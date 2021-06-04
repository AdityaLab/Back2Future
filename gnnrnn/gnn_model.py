import dgl
import dgl.nn as dnn
import numpy as np
from scipy.sparse.construct import rand, random
import torch as th
import torch.nn as nn
from torch.nn.modules.activation import Tanh


class GNNRNN(nn.Module):
    """
    GNNRNN module to predict the revisions
    """

    def __init__(
        self, num_feats: int, rnn_hidden: int, heads: int = 1, device: str = "cuda"
    ):
        """
        :param num_feats: no. of features
        :param rnn_hidden: dimensions of hidden states
        :param heads: heads of attention in GatedGNN
        :param device: cpu or cuda
        """
        super(GNNRNN, self).__init__()
        assert rnn_hidden % heads == 0
        self.rnn_hidden = rnn_hidden
        self.num_feats = num_feats
        # GNN layer
        self.gnn = dnn.GATConv(rnn_hidden, rnn_hidden // heads, heads)
        # RNN for each feature
        self.rnns = [nn.GRUCell(1, rnn_hidden).to(device) for _ in range(num_feats)]
        # RNN hidden state decoder for each feature
        self.decoders = [nn.Linear(rnn_hidden, 1).to(device) for _ in range(num_feats)]
        self.device = device

    def forward_step2(
        self, g: dgl.DGLGraph, batch_vec: th.Tensor, hidden_vec: th.Tensor
    ):
        r"""
        Input
        -------

        :param g: DGLGraph with "feature" number of nodes
        :param batch_vec: vector of shape [batch, features]
        :param hidden_vec: vector of shape [features, batch, rnn_hidden]

        Outputs:
        ans: Revised prediction of shape: [batch, features]
        new_hidden: new hidden rep of shape [features, batch, rnn_hidden]
        """
        hidden = []
        ans = []
        for i in range(self.num_feats):
            hidden.append(self.rnns[i](batch_vec[:, i].unsqueeze(-1), hidden_vec[i]))
        hidden = th.stack(hidden)  # Shape: [features, batch, rnn_hidden]
        new_hidden = []
        for i in range(hidden_vec.shape[1]):
            new_hidden.append(
                self.gnn(g, hidden[:, i]).reshape(self.num_feats, self.rnn_hidden)
            )
        new_hidden = th.stack(new_hidden)  # Shape: [batch, features, rnn_hidden]
        new_hidden = new_hidden.transpose(0, 1)
        for i in range(self.num_feats):
            ans.append(self.decoders[i](hidden[i]).squeeze(-1))
        ans = th.stack(ans)  # Shape: [features, batch]
        ans = ans.transpose(1, 0)
        return ans, new_hidden
    
    def forward_step(
        self, g: dgl.DGLGraph, batch_vec: th.Tensor, hidden_vec: th.Tensor
    ):
        r"""
        Input
        -------

        :param g: DGLGraph with "feature" number of nodes
        :param batch_vec: vector of shape [batch, features]
        :param hidden_vec: vector of shape [features, batch, rnn_hidden]

        Outputs:
        ans: Revised prediction of shape: [batch, features]
        new_hidden: new hidden rep of shape [features, batch, rnn_hidden]
        """
        hidden = []
        ans = []
        for i in range(self.num_feats):
            hidden.append(self.rnns[i](batch_vec[:, i].unsqueeze(-1), hidden_vec[i]))
        hidden = th.stack(hidden).transpose(1,0)  # Shape: [batch, features, rnn_hidden]
        g1 = dgl.batch([g]*hidden_vec.shape[1])
        new_hidden = self.gnn(g1, hidden.reshape(hidden.shape[0]*hidden.shape[1], -1)).reshape(hidden_vec.shape[1], self.num_feats, -1) # Shape: [batch, features, rnn_hidden]
        new_hidden = new_hidden.transpose(0, 1)
        hidden = hidden.transpose(1,0)  # Shape: [batch, features, rnn_hidden]
        for i in range(self.num_feats):
            ans.append(self.decoders[i](hidden[i]).squeeze(-1))
        ans = th.stack(ans)  # Shape: [features, batch]
        ans = ans.transpose(1, 0)
        return ans, new_hidden

    def forward(self, g_: dgl.DGLGraph, history_stack: th.Tensor, teach: bool = True, prob:float = 0.5):
        r"""
        Inputs
        ------

        :param g: DGLGraph with "feature" number of nodes
        :param history_stack: revision history of shape [batch, time, features]

        Output
        ------
        ans: Out prediction of shape [batch, time, features]
        hiddens: Hidden stats of shape [batch, time, features, rnn_hidden]
        """
        g = dgl.add_self_loop(g_)
        hidden_t = th.stack(
            [
                th.zeros(history_stack.shape[0], self.rnn_hidden).to(self.device)
                for _ in range(self.num_feats)
            ]
        )
        ans = []
        hiddens = []
        for t in range(history_stack.shape[1]):
            if teach or t == 0:
                ans_t, hidden_t = self.forward_step(g, history_stack[:, t], hidden_t)
            elif np.random.rand()<prob:
                ans_t, hidden_t = self.forward_step(g, ans_t.detach(), hidden_t)
            ans.append(ans_t)
            hiddens.append(hidden_t)
        ans = th.stack(ans)
        ans = ans.transpose(1, 0)
        hiddens = th.stack(hiddens)  # SHape: [time, features, batch, rnn_hidden]
        hiddens = hiddens.transpose(2, 1).transpose(
            1, 0
        )  # Shape: [batch, time, features, rnn_hidden]
        return ans, hiddens

class GNNRNNv2(nn.Module):
    """
    GNNRNN module to predict the revisions
    """

    def __init__(
        self, num_feats: int, rnn_hidden: int, heads: int = 1, device: str = "cuda"
    ):
        """
        :param num_feats: no. of features
        :param rnn_hidden: dimensions of hidden states
        :param heads: heads of attention in GatedGNN
        :param device: cpu or cuda
        """
        super(GNNRNNv2, self).__init__()
        assert rnn_hidden % heads == 0
        self.rnn_hidden = rnn_hidden
        self.num_feats = num_feats
        self.encoders = [nn.Linear(1, rnn_hidden) for _ in range(num_feats)]
        # GNN layer
        self.gnn = dnn.GATConv(rnn_hidden, rnn_hidden // heads, heads)
        # RNN for each feature
        self.rnns = [nn.GRU(rnn_hidden, rnn_hidden).to(device) for _ in range(num_feats)]
        # RNN hidden state decoder for each feature
        self.decoders = [nn.Linear(rnn_hidden, 1).to(device) for _ in range(num_feats)]
        self.device = device


    def forward(self, g: dgl.DGLGraph, history_stack: th.Tensor, teach: bool = True):
        r"""
        Inputs
        ------

        :param g: DGLGraph with "feature" number of nodes
        :param history_stack: revision history of shape [batch, time, features]

        Output
        ------
        ans: Out prediction of shape [batch, time, features]
        hiddens: Hidden stats of shape [batch, time, features, rnn_hidden]
        """
        g = dgl.add_self_loop(g)
        enc = []
        for n in range(self.num_feats):
            enc.append(self.encoders[n](history_stack[:, :, n].unsqueeze(-1)))
        enc = th.stack(enc, dim=-2) #shape: [batch, time, features, rnn_hidden]
        gnn_encodes = []
        g1 = dgl.batch([g]*enc.shape[0]*enc.shape[1])
        gnn_encodes = self.gnn(g1, enc.reshape(enc.shape[0]*enc.shape[1]*enc.shape[2], enc.shape[3]))
        gnn_encodes = gnn_encodes.reshape(enc.shape[0],enc.shape[1],enc.shape[2],-1)

        hiddens = []
        for f in range(gnn_encodes.shape[2]):
            hiddens.append(self.rnns[f](gnn_encodes[:, :, f, :].transpose(1,0))[0])
        hiddens = th.stack(hiddens, dim=2).transpose(1,0) # Shape: [batch, time, features, rnn_hidden]
        ans = []
        for f in range(gnn_encodes.shape[2]):
            ans.append(self.decoders[f](hiddens[:,:,f,:]))
        ans = th.stack(ans, dim=2).squeeze(-1)

        return ans, hiddens



class TransformerAttn(nn.Module):
    """
    Module that calculates self-attention weights using transformer like attention
    """

    def __init__(self, dim_in=40, value_dim=40, key_dim=40) -> None:
        """
        param dim_in: Dimensionality of input sequence
        param value_dim: Dimension of value transform
        param key_dim: Dimension of key transform
        """
        super(TransformerAttn, self).__init__()
        self.value_layer = nn.Linear(dim_in, value_dim)
        self.query_layer = nn.Linear(dim_in, value_dim)
        self.key_layer = nn.Linear(dim_in, key_dim)

    def forward(self, seq):
        """
        param seq: Sequence in dimension [Seq len, Batch, Hidden size]

        Output:
        ans: shape [Seq len, Batch, Hidden size]
        """
        seq_in = seq.transpose(0, 1)
        value = self.value_layer(seq_in)
        query = self.query_layer(seq_in)
        keys = self.key_layer(seq_in)
        weights = (value @ query.transpose(1, 2)) / np.sqrt(
            seq.shape[-1]
        )  # Shape: [Batch, Seqlen, Seqlen]
        weights = th.softmax(weights, -1)  # Shape: [Batch, Seqlen, SeqLen]
        return (weights @ keys).transpose(1, 0)  # Shape [SeqLen, Batch, Hidden Size]


class MainFNNNetwork(nn.Module):
    """
    Main Network for DeepCovid
    """

    def __init__(
        self,
        num_features,
        revise_hidden_dims,
        layers,
        final_layers,
        dropout=0.0,
        attention=False,
        activation=nn.LeakyReLU,
        device="cuda",
    ):
        super(MainFNNNetwork, self).__init__()
        self.num_features = num_features
        self.revise_hidden_dims = revise_hidden_dims
        self.layers = layers
        self.final_layers = final_layers
        self.dropout = dropout
        self.attention = attention
        self.activation = activation
        self.device = device

        self.encode_inp = [
            nn.Linear(num_features, layers[0]),
            nn.Dropout(self.dropout),
            self.activation(),
        ]
        for l in range(1, len(layers)):
            self.encode_inp.extend(
                [
                    nn.Linear(layers[l - 1], layers[l]),
                    nn.Dropout(self.dropout),
                    self.activation(),
                ]
            )
        self.encode_inp = nn.Sequential(*self.encode_inp)
        if attention:
            self.hidden_module = TransformerAttn(
                revise_hidden_dims * num_features,
                final_layers[0] - layers[-1],
                final_layers[0] - layers[-1],
            )
        else:
            self.hidden_module = nn.Linear(
                revise_hidden_dims * num_features, final_layers[0] - layers[-1]
            )

        self.final_mod = []
        for l in range(1, len(final_layers)):
            self.final_mod.extend(
                [
                    nn.Linear(final_layers[l - 1], final_layers[l]),
                    nn.Dropout(self.dropout),
                    self.activation(),
                ]
            )
        self.final_mod.append(nn.Linear(final_layers[-1], 1))
        self.final_mod = nn.Sequential(*self.final_mod)

    def forward(self, inp_feats, revision_hidden):
        r"""
        Inputs
        ----

        :param inp_feats: input feature shape: [batch, features]
        :param revision_hidden: shape [batch, time, features, rnn_hidden]

        Outputs
        ----
        dec: SHape [batch, 1]
        """
        enc1 = self.encode_inp(inp_feats)
        enc2 = revision_hidden.reshape(
            revision_hidden.shape[0],
            revision_hidden.shape[1],
            revision_hidden.shape[2] * revision_hidden.shape[3],
        )
        if self.attention:
            enc2 = self.hidden_module(enc2.transpose(0, 1))[-1]
        else:
            enc2 = self.hidden_module(enc2[:, -1])
        enc = th.cat([enc1, enc2], dim=-1)
        dec = self.final_mod(enc)
        return dec


class Refinerv1(nn.Module):
    """
    Refiner Architecture.
    Takes History embed and GNNRNN embed.
    """

    def __init__(
        self,
        num_features,
        revise_hidden_dims,
        pred_dim,
        layers,
        final_layers,
        dropout=0.0,
        attention=False,
        activation=nn.LeakyReLU,
        device="cuda",
    ):
        super(Refiner, self).__init__()
        self.num_features = num_features
        self.revise_hidden_dims = revise_hidden_dims
        self.layers = layers
        self.final_layers = final_layers
        self.dropout = dropout
        self.attention = attention
        self.activation = activation
        self.pred_dim = pred_dim
        self.device = device

        self.pred_inp = [
            nn.Linear(revise_hidden_dims, layers[0]),
            nn.Dropout(self.dropout),
            self.activation(),
        ]
        for l in range(1, len(layers)):
            self.pred_inp.extend(
                [
                    nn.Linear(layers[l - 1], layers[l]),
                    nn.Dropout(self.dropout),
                    self.activation(),
                ]
            )
        self.pred_inp = nn.Sequential(*self.pred_inp)
        if attention:
            self.hidden_module = TransformerAttn(
                revise_hidden_dims * num_features,
                final_layers[0] - layers[-1],
                final_layers[0] - layers[-1],
            )
        else:
            self.hidden_module = nn.Linear(
                revise_hidden_dims * num_features, final_layers[0] - layers[-1]
            )

        self.final_mod = []
        for l in range(1, len(final_layers)):
            self.final_mod.extend(
                [
                    nn.Linear(final_layers[l - 1], final_layers[l]),
                    nn.Dropout(self.dropout),
                    self.activation(),
                ]
            )
        self.final_mod.append(nn.Linear(final_layers[-1], 1))
        self.final_mod.append(nn.Tanh())
        self.final_mod = nn.Sequential(*self.final_mod)

    def forward(self, pred_embed, revision_hidden):
        r"""
        Inputs
        ----

        :param inp_feats: input feature shape: [batch, features]
        :param revision_hidden: shape [batch, time, features, rnn_hidden]

        Outputs
        ----
        dec: SHape [batch, 1]
        """
        enc1 = self.pred_inp(pred_embed)
        enc2 = revision_hidden.reshape(
            revision_hidden.shape[0],
            revision_hidden.shape[1],
            revision_hidden.shape[2] * revision_hidden.shape[3],
        )
        if self.attention:
            enc2 = self.hidden_module(enc2.transpose(0, 1))[-1]
        else:
            enc2 = self.hidden_module(enc2[:, -1])
        enc = th.cat([enc1, enc2], dim=-1)
        dec = self.final_mod(enc)
        return dec


class CAttention(nn.Module):
    def __init__(self, value_dim, key_dim, hidden_dim=None):
        super(CAttention, self).__init__()
        if hidden_dim is None:
            hidden_dim = value_dim
        self.val_enc, self.key_enc = (
            nn.Linear(value_dim, hidden_dim),
            nn.Linear(key_dim, hidden_dim),
        )

    def forward(self, value, keys):
        """
        :param value: shape [Batch, val_dim]
        :param keys: shape [Batch, Seq, key_dim]
        """
        val, k = self.val_enc(value), self.key_enc(keys)
        atten = (val[:, None, :] * k).sum(-1)  # shape: [Batch, Seq]
        atten = th.softmax(atten, dim=-1)
        out = (keys * atten.unsqueeze(-1)).sum(-2)
        return out, atten


class Refinerv2(nn.Module):
    """
    Refiner Architecture.
    Takes History embed and GNNRNN embed.
    """

    def __init__(
        self,
        num_features,
        revise_hidden_dims,
        pred_dim,
        layers,
        final_layers,
        dropout=0.0,
        attention=False,
        activation=nn.LeakyReLU,
        device="cuda",
    ):
        super(Refinerv2, self).__init__()
        self.num_features = num_features
        self.revise_hidden_dims = revise_hidden_dims
        self.layers = layers
        self.final_layers = final_layers
        self.dropout = dropout
        self.attention = attention
        self.activation = activation
        self.pred_dim = pred_dim
        self.device = device

        self.pred_inp = [
            nn.Linear(pred_dim, layers[0]),
            nn.Dropout(self.dropout),
            self.activation(),
        ]
        for l in range(1, len(layers)):
            self.pred_inp.extend(
                [
                    nn.Linear(layers[l - 1], layers[l]),
                    nn.Dropout(self.dropout),
                    self.activation(),
                ]
            )
        self.pred_inp = nn.Sequential(*self.pred_inp)

        self.hidden_module = CAttention(layers[-1], revise_hidden_dims)

        self.final_mod = [nn.Linear(layers[-1] + revise_hidden_dims, final_layers[0])]
        for l in range(1, len(final_layers)):
            self.final_mod.extend(
                [
                    nn.Linear(final_layers[l - 1], final_layers[l]),
                    nn.Dropout(self.dropout),
                    self.activation(),
                ]
            )
        self.final_mod.append(nn.Linear(final_layers[-1], 1))
        self.final_mod.append(nn.Tanh())
        self.final_mod = nn.Sequential(*self.final_mod)

    def forward(self, pred_embed, revision_hidden):
        r"""
        Inputs
        ----

        :param inp_feats: input feature shape: [batch, features]
        :param revision_hidden: shape [batch, time, features, rnn_hidden]

        Outputs
        ----
        dec: SHape [batch, 1]
        """
        enc1 = self.pred_inp(pred_embed)

        enc2 = revision_hidden.sum(-2)
        enc3, atten = self.hidden_module(enc1, enc2)

        enc = th.cat([enc1, enc3], dim=-1)
        dec = self.final_mod(enc)
        return dec, atten


class Refiner(nn.Module):
    """
    Refiner Architecture.
    Takes History embed and GNNRNN embed.
    """

    def __init__(
        self,
        num_features,
        revise_hidden_dims,
        bias_dim,
        atten_dim=50,
        revise_out_dim=50,
        bias_out_dim=50,
        out_hidden=50,
        device="cuda",
    ):
        super(Refiner, self).__init__()
        self.num_features = num_features
        self.revise_hidden_dims = revise_hidden_dims

        self.atten_dim = atten_dim
        self.revise_atten = nn.Linear(revise_hidden_dims, atten_dim)
        self.pred_atten = nn.Linear(1, atten_dim)
        self.pred_inp = nn.Linear(1, atten_dim)
        self.out_rev = nn.Linear(revise_hidden_dims, revise_out_dim)
        self.bias_layer = nn.Linear(bias_dim, bias_out_dim)
        self.out_layers = nn.Sequential(
            nn.Linear(atten_dim+revise_out_dim+bias_out_dim, out_hidden),
            nn.ReLU(),
            nn.Linear(out_hidden, 1),
            nn.Tanh()
        )


    def forward(self, bias_embed, revision_hidden, curr_pred):
        r"""
        Inputs
        ----
        :param bias_embed: model_bias shape: [batch, hidden]
        :param revision_hidden: shape [batch, time, features, rnn_hidden]
        :param curr_pred: input feature shape: [batch]

        Outputs
        ----
        dec: SHape [batch]
        """
        # Attention on revision
        revise_atten = self.revise_atten(revision_hidden[:,-1,:,:])
        pred_atten = self.pred_atten(curr_pred.unsqueeze(-1))
        pred_inp = self.pred_inp(curr_pred.unsqueeze(-1))
        atten_weights = th.matmul(revise_atten, pred_atten.unsqueeze(-1)).squeeze(-1)
        atten_weights = th.softmax(atten_weights, -1)
        sum_revise = (self.out_rev(revision_hidden[:,-1,:,:]) * atten_weights.unsqueeze(-1)).sum(-2) # batch, rnn_hidden
        bias_in = self.bias_layer(bias_embed)
        inp = th.cat([pred_inp, sum_revise, bias_in], -1)
        out = self.out_layers(inp)
        return out.squeeze(-1)


class ModelBiasEncoder(nn.Module):
    def __init__(self, out_dim, in_dim=3, bidirectional=True, dropout=0.0):
        super(ModelBiasEncoder, self).__init__()
        self.out_dim = out_dim // (2 if bidirectional else 1)
        self.rnn = nn.GRU(
            in_dim, self.out_dim, dropout=dropout, bidirectional=bidirectional
        )

    def forward(self, pred_stack):
        """
        :param pred_stack: shape [Seq, 3]
        """
        if len(pred_stack.shape)==2:
            inp = pred_stack.unsqueeze(1)
        else:
            inp = pred_stack
        out, _ = self.rnn(inp)
        if len(pred_stack.shape)==2:
            return out.squeeze(1)
        return out


def test():
    od = 10
    st = th.rand(5, 3)
    model = ModelBiasEncoder(od)
    ans = model(st)
    return ans

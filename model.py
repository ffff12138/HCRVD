<<<<<<< HEAD
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from itertools import repeat
import numpy as np

class SpatialDropout(nn.Module):
    def __init__(self, drop=0.5):
        super(SpatialDropout, self).__init__()
        self.drop = drop

    def forward(self, inputs, noise_shape=None):
        outputs = inputs.clone()
        if noise_shape is None:
            noise_shape = (inputs.shape[0], *repeat(1, inputs.dim() - 2), inputs.shape[-1])
        self.noise_shape = noise_shape
        if not self.training or self.drop == 0:
            return inputs
        else:
            noises = self._make_noises(inputs)
            if self.drop == 1:
                noises.fill_(0.0)
            else:
                noises.bernoulli_(1 - self.drop).div_(1 - self.drop)
            noises = noises.expand_as(inputs)
            outputs.mul_(noises)
            return outputs

    def _make_noises(self, inputs):
        return inputs.new().resize_(self.noise_shape)

class TGA(nn.Module):
    def __init__(self, vocab_size, embedding_dim, encode_dim, batch_size, use_gpu, pretrained_weight=None):
        super(TGA, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encode_dim = encode_dim
        self.hidden_dim = encode_dim
        self.num_layers=1
        self.embedding_dim=embedding_dim
        self.gru= nn.GRU(self.embedding_dim, self.hidden_dim)
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.node_list = []
        self.sdrop = SpatialDropout(0.2)
        torch.cuda.set_device("cuda:0")
        self.th = torch.cuda if use_gpu else torch
        self.batch_node = None
        self.tanh=torch.nn.Tanh()
        if pretrained_weight is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))

    def create_tensor(self, tensor):
        if self.use_gpu:
            return tensor.to("cuda:0")
        return tensor

    def traverse_mul(self, node, batch_index):
        size = len(node)
        if not size:
            return None
        batch_current = self.create_tensor(Variable(torch.zeros(size, self.embedding_dim)))

        sub_tree_node_list = []
        sub_tree_hid_list = []
        index, children_index = [], []
        current_node, children = [], []
        for i in range(size):
            if node[i][0] != -1:
                index.append(i)
                current_node.append(node[i][0])
                temp = node[i][1:]
                c_num = len(temp)
                for j in range(c_num):
                    if temp[j][0] != -1:
                        if len(children_index) <= j:
                            children_index.append([i])
                            children.append([temp[j]])
                        else:
                            children_index[j].append(i)
                            children[j].append(temp[j])
            else:
                print(node[i])
                batch_index[i] = -1

        batch_current = self.sdrop(batch_current.index_copy(0, Variable(self.th.LongTensor(index)),
                                                 self.embedding(Variable(self.th.LongTensor(current_node)))))

        children_hide = self.create_tensor(Variable(torch.zeros(self.num_layers, size, self.encode_dim)))
        for c in range(len(children)):
            hid_zero = self.create_tensor(Variable(torch.zeros(self.num_layers, size, self.hidden_dim)))
            batch_children_index = [batch_index[i] for i in children_index[c]]
            tree, hid = self.traverse_mul(children[c], batch_children_index)
            if tree is not None:
                children_hide += hid_zero.index_copy(1, Variable(self.th.LongTensor(children_index[c])),hid)

        batch_current, hidden_state = self.gru(batch_current.unsqueeze(0), children_hide)
        batch_current, hidden_state = batch_current.squeeze(0), hidden_state

        batch_index = [i for i in batch_index if i != -1]
        b_in = Variable(self.th.LongTensor(batch_index))
        self.node_list.append(self.batch_node.index_copy(0, b_in, batch_current))
        return batch_current,hidden_state

    def attention_net(self, lstm_output, final_state):
        lstm_output=self.tanh(lstm_output.transpose(0,1))
        hidden = final_state.squeeze(0).unsqueeze(2)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context

    def forward(self, x, bs):
        self.batch_size = bs
        self.batch_node = self.create_tensor(Variable(torch.zeros(self.batch_size, self.encode_dim)))
        self.node_list = []
        final_state ,final_hidden_state=self.traverse_mul(x, list(range(self.batch_size)))
        self.node_list = torch.stack(self.node_list)
        attn_output = self.attention_net(self.node_list,final_hidden_state)
        return attn_output


class Propogator(nn.Module):
    def __init__(self, state_dim):
        super(Propogator, self).__init__()

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid(),
            nn.Dropout(0.5)
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid(),
            nn.Dropout(0.5)
        )
        self.tansform = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Tanh()
        )

    def forward(self, state_in,state_out,state_cur, A):
        n_node=A.shape[1]
        A_in = A[:, :, :n_node]
        A_out = A[:, :, n_node:]
        a_in = torch.bmm(A_in, state_in)
        a_out = torch.bmm(A_out, state_out)
        a = torch.cat((a_in, a_out, state_cur), 2)
        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_in, a_out, r * state_cur), 2)
        h_hat = self.tansform(joined_input)
        output = (1 - z) * state_cur + z * h_hat
        return output

class GGA(nn.Module):
    def __init__(self,state_dim,annotation_dim,n_steps):
        super(GGA, self).__init__()
        self.state_dim = state_dim
        self.n_steps = n_steps
        self.annotation_dim=annotation_dim
        self.in_fc = nn.Linear(self.state_dim, self.state_dim)
        self.out_fc = nn.Linear(self.state_dim, self.state_dim)
        self.propogator = Propogator(self.state_dim)
        self.attention = nn.Sequential(
            nn.Linear(self.state_dim + self.annotation_dim, 1),
            nn.Sigmoid()
        )
        self.out = nn.Sequential(
            nn.Linear(self.state_dim + self.annotation_dim, 1 ),
            nn.Tanh()
        )
        self.result = nn.Sigmoid()
        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, prop_state, annotation, A):
        for i_step in range(self.n_steps):
            in_state=self.in_fc(prop_state)
            out_state=self.out_fc(prop_state)
            prop_state = self.propogator(in_state, out_state, prop_state, A)
        join_state = torch.cat((prop_state, annotation), 2)
        atten = self.attention(join_state)
        ou = self.out(join_state)
        mul = atten * ou
        mul = mul.squeeze(2)
        w_sum = torch.sum(mul, dim=1)
        res = self.result(w_sum)
        return res


class TGGA(nn.Module):
    def __init__(self, embedding_dim, vocab_size, encode_dim, batch_size, use_gpu=True, pretrained_weight=None):
        super(TGGA, self).__init__()
        self.gpu = use_gpu
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.encode_dim = encode_dim
        self.encoder = TGA(self.vocab_size, self.embedding_dim, self.encode_dim,
                                        self.batch_size, self.gpu, pretrained_weight)
        self.gga=GGA(encode_dim,encode_dim,4)

    def create_adjacency_matrix(self,adj):
        adj = np.hstack([adj, np.transpose(adj)])
        return adj

    def padadj(self, adj, maxlen):
        n = len(adj)
        adj[adj > 1] = 1
        if n >= maxlen:
            return adj[:maxlen, :maxlen]
        else:
            padded_adj = np.zeros((maxlen, maxlen))
            padded_adj[:n, :n] = adj
            return padded_adj

    def get_zeros(self, num):
        zeros = Variable(torch.zeros(num, self.encode_dim))
        if self.gpu:
            return zeros.to("cuda:0")
        return zeros

    def forward(self, x,adj):
        lens = [len(item) for item in x]
        max_len = max(lens)
        encodes = []
        for i in range(self.batch_size):
            for j in range(lens[i]):
                encodes.append(x[i][j])
        encodes = self.encoder(encodes, sum(lens))
        seq, start, end ,full_adj= [], 0, 0,[]
        for i in range(self.batch_size):
            end += lens[i]
            seq.append(encodes[start:end])
            full_adj.append(self.create_adjacency_matrix(self.padadj(adj[i], max_len)))
            if max_len-lens[i]:
                seq.append(self.get_zeros(max_len-lens[i]))
            start = end
        encodes = torch.cat(seq)
        encodes = encodes.view(self.batch_size, max_len, -1)
        full_adj = torch.tensor(np.array(full_adj)).to(torch.float32).to('cuda:0')
        out=self.gga(encodes,encodes,full_adj)
        return out
=======
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from itertools import repeat
import numpy as np

class SpatialDropout(nn.Module):
    def __init__(self, drop=0.5):
        super(SpatialDropout, self).__init__()
        self.drop = drop

    def forward(self, inputs, noise_shape=None):
        outputs = inputs.clone()
        if noise_shape is None:
            noise_shape = (inputs.shape[0], *repeat(1, inputs.dim() - 2), inputs.shape[-1])
        self.noise_shape = noise_shape
        if not self.training or self.drop == 0:
            return inputs
        else:
            noises = self._make_noises(inputs)
            if self.drop == 1:
                noises.fill_(0.0)
            else:
                noises.bernoulli_(1 - self.drop).div_(1 - self.drop)
            noises = noises.expand_as(inputs)
            outputs.mul_(noises)
            return outputs

    def _make_noises(self, inputs):
        return inputs.new().resize_(self.noise_shape)

class TGA(nn.Module):
    def __init__(self, vocab_size, embedding_dim, encode_dim, batch_size, use_gpu, pretrained_weight=None):
        super(TGA, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encode_dim = encode_dim
        self.hidden_dim = encode_dim
        self.num_layers=1
        self.embedding_dim=embedding_dim
        self.gru= nn.GRU(self.embedding_dim, self.hidden_dim)
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.node_list = []
        self.sdrop = SpatialDropout(0.2)
        torch.cuda.set_device("cuda:0")
        self.th = torch.cuda if use_gpu else torch
        self.batch_node = None
        self.tanh=torch.nn.Tanh()
        if pretrained_weight is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))

    def create_tensor(self, tensor):
        if self.use_gpu:
            return tensor.to("cuda:0")
        return tensor

    def traverse_mul(self, node, batch_index):
        size = len(node)
        if not size:
            return None
        batch_current = self.create_tensor(Variable(torch.zeros(size, self.embedding_dim)))

        sub_tree_node_list = []
        sub_tree_hid_list = []
        index, children_index = [], []
        current_node, children = [], []
        for i in range(size):
            if node[i][0] != -1:
                index.append(i)
                current_node.append(node[i][0])
                temp = node[i][1:]
                c_num = len(temp)
                for j in range(c_num):
                    if temp[j][0] != -1:
                        if len(children_index) <= j:
                            children_index.append([i])
                            children.append([temp[j]])
                        else:
                            children_index[j].append(i)
                            children[j].append(temp[j])
            else:
                print(node[i])
                batch_index[i] = -1

        batch_current = self.sdrop(batch_current.index_copy(0, Variable(self.th.LongTensor(index)),
                                                 self.embedding(Variable(self.th.LongTensor(current_node)))))

        children_hide = self.create_tensor(Variable(torch.zeros(self.num_layers, size, self.encode_dim)))
        for c in range(len(children)):
            hid_zero = self.create_tensor(Variable(torch.zeros(self.num_layers, size, self.hidden_dim)))
            batch_children_index = [batch_index[i] for i in children_index[c]]
            tree, hid = self.traverse_mul(children[c], batch_children_index)
            if tree is not None:
                children_hide += hid_zero.index_copy(1, Variable(self.th.LongTensor(children_index[c])),hid)

        batch_current, hidden_state = self.gru(batch_current.unsqueeze(0), children_hide)
        batch_current, hidden_state = batch_current.squeeze(0), hidden_state

        batch_index = [i for i in batch_index if i != -1]
        b_in = Variable(self.th.LongTensor(batch_index))
        self.node_list.append(self.batch_node.index_copy(0, b_in, batch_current))
        return batch_current,hidden_state

    def attention_net(self, lstm_output, final_state):
        lstm_output=self.tanh(lstm_output.transpose(0,1))
        hidden = final_state.squeeze(0).unsqueeze(2)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context

    def forward(self, x, bs):
        self.batch_size = bs
        self.batch_node = self.create_tensor(Variable(torch.zeros(self.batch_size, self.encode_dim)))
        self.node_list = []
        final_state ,final_hidden_state=self.traverse_mul(x, list(range(self.batch_size)))
        self.node_list = torch.stack(self.node_list)
        attn_output = self.attention_net(self.node_list,final_hidden_state)
        return attn_output


class Propogator(nn.Module):
    def __init__(self, state_dim):
        super(Propogator, self).__init__()

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid(),
            nn.Dropout(0.5)
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid(),
            nn.Dropout(0.5)
        )
        self.tansform = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Tanh()
        )

    def forward(self, state_in,state_out,state_cur, A):
        n_node=A.shape[1]
        A_in = A[:, :, :n_node]
        A_out = A[:, :, n_node:]
        a_in = torch.bmm(A_in, state_in)
        a_out = torch.bmm(A_out, state_out)
        a = torch.cat((a_in, a_out, state_cur), 2)
        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_in, a_out, r * state_cur), 2)
        h_hat = self.tansform(joined_input)
        output = (1 - z) * state_cur + z * h_hat
        return output

class GGA(nn.Module):
    def __init__(self,state_dim,annotation_dim,n_steps):
        super(GGA, self).__init__()
        self.state_dim = state_dim
        self.n_steps = n_steps
        self.annotation_dim=annotation_dim
        self.in_fc = nn.Linear(self.state_dim, self.state_dim)
        self.out_fc = nn.Linear(self.state_dim, self.state_dim)
        self.propogator = Propogator(self.state_dim)
        self.attention = nn.Sequential(
            nn.Linear(self.state_dim + self.annotation_dim, 1),
            nn.Sigmoid()
        )
        self.out = nn.Sequential(
            nn.Linear(self.state_dim + self.annotation_dim, 1 ),
            nn.Tanh()
        )
        self.result = nn.Sigmoid()
        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, prop_state, annotation, A):
        for i_step in range(self.n_steps):
            in_state=self.in_fc(prop_state)
            out_state=self.out_fc(prop_state)
            prop_state = self.propogator(in_state, out_state, prop_state, A)
        join_state = torch.cat((prop_state, annotation), 2)
        atten = self.attention(join_state)
        ou = self.out(join_state)
        mul = atten * ou
        mul = mul.squeeze(2)
        w_sum = torch.sum(mul, dim=1)
        res = self.result(w_sum)
        return res


class TGGA(nn.Module):
    def __init__(self, embedding_dim, vocab_size, encode_dim, batch_size, use_gpu=True, pretrained_weight=None):
        super(TGGA, self).__init__()
        self.gpu = use_gpu
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.encode_dim = encode_dim
        self.encoder = TGA(self.vocab_size, self.embedding_dim, self.encode_dim,
                                        self.batch_size, self.gpu, pretrained_weight)
        self.gga=GGA(encode_dim,encode_dim,4)

    def create_adjacency_matrix(self,adj):
        adj = np.hstack([adj, np.transpose(adj)])
        return adj

    def padadj(self, adj, maxlen):
        n = len(adj)
        adj[adj > 1] = 1
        if n >= maxlen:
            return adj[:maxlen, :maxlen]
        else:
            padded_adj = np.zeros((maxlen, maxlen))
            padded_adj[:n, :n] = adj
            return padded_adj

    def get_zeros(self, num):
        zeros = Variable(torch.zeros(num, self.encode_dim))
        if self.gpu:
            return zeros.to("cuda:0")
        return zeros

    def forward(self, x,adj):
        lens = [len(item) for item in x]
        max_len = max(lens)
        encodes = []
        for i in range(self.batch_size):
            for j in range(lens[i]):
                encodes.append(x[i][j])
        encodes = self.encoder(encodes, sum(lens))
        seq, start, end ,full_adj= [], 0, 0,[]
        for i in range(self.batch_size):
            end += lens[i]
            seq.append(encodes[start:end])
            full_adj.append(self.create_adjacency_matrix(self.padadj(adj[i], max_len)))
            if max_len-lens[i]:
                seq.append(self.get_zeros(max_len-lens[i]))
            start = end
        encodes = torch.cat(seq)
        encodes = encodes.view(self.batch_size, max_len, -1)
        full_adj = torch.tensor(np.array(full_adj)).to(torch.float32).to('cuda:0')
        out=self.gga(encodes,encodes,full_adj)
        return out
>>>>>>> 00b7fc3 (Signend-off-by: Song <ssong12138@163.com>)

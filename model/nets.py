import math
import os
import torch
import numpy as np
from torch.nn.init import normal_ as normal_init
import torch.nn.functional as F
import torch.nn as nn
import random
from torch.distributions import uniform
from .rotary_embedding import RotaryEmbedding
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
__all__ = ['Cbox4CR']


class MultilayerNN(nn.Module):
    # torch min will just return a single element in the tensor
    def __init__(self, center_dim, offset_dim):
        super(MultilayerNN, self).__init__()
        self.center_dim = center_dim
        self.offset_dim = offset_dim

        expand_dim = center_dim * 2

        self.mats1_center = nn.Parameter(torch.FloatTensor(center_dim, expand_dim))
        nn.init.xavier_uniform_(self.mats1_center)
        self.register_parameter("mats1_center", self.mats1_center)

        self.mats1_offset = nn.Parameter(torch.FloatTensor(center_dim, expand_dim))
        nn.init.xavier_uniform_(self.mats1_offset)
        self.register_parameter("mats1_offset", self.mats1_offset)

        self.post_mats_center = nn.Parameter(torch.FloatTensor(expand_dim * 2, center_dim))
        # every time the initial is different
        nn.init.xavier_uniform_(self.post_mats_center)
        self.register_parameter("post_mats_center", self.post_mats_center)

        self.post_mats_offset = nn.Parameter(torch.FloatTensor(expand_dim * 2, center_dim))
        # every time the initial is different
        nn.init.xavier_uniform_(self.post_mats_offset)
        self.register_parameter("post_mats_offset", self.post_mats_offset)

    def forward(self, center_emb, offset_emb):
        # if A and B are of shape (3, 4), torch.cat([A, B], dim=0) will be of shape (6, 4)
        # and torch.stack([A, B], dim=0) will be of shape (2, 3, 4).
        temp1 = F.relu(torch.matmul(center_emb, self.mats1_center))
        temp2 = F.relu(torch.matmul(offset_emb, self.mats1_offset))

        temp3 = torch.cat([temp1, temp2], dim=-1)

        out_center = torch.matmul(temp3, self.post_mats_center)
        out_offset = torch.matmul(temp3, self.post_mats_offset)

        return out_center, out_offset


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class CenterIntersection(nn.Module):

    def __init__(self, dim):
        super(CenterIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        layer1_act = F.relu(self.layer1(embeddings))  # (num_conj, dim)
        attention = F.softmax(self.layer2(layer1_act), dim=0)  # (num_conj, dim)
        embedding = torch.sum(attention * embeddings, dim=0)

        return embedding


class BoxOffsetIntersection(nn.Module):

    def __init__(self, dim):
        super(BoxOffsetIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        layer1_act = F.relu(self.layer1(embeddings))
        layer1_mean = torch.mean(layer1_act, dim=0)
        gate = torch.sigmoid(self.layer2(layer1_mean))
        offset, _ = torch.min(embeddings, dim=0)

        return offset * gate


class Mlp(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(Mlp, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.layer1 = nn.Linear(self.input_dim, self.hidden_dim)  # 1st layer
        self.layer0 = nn.Linear(self.hidden_dim, self.output_dim)  # final layer
        for nl in range(2, num_layers + 1):
            setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)

    def forward(self, embedding):
        x = embedding
        for nl in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, "layer{}".format(nl))(x))
        x = self.layer0(x)
        return x


def Identity(x):
    return x


class Box:
    def __init__(self, min_embed, max_embed):
        self.min_embed = min_embed
        self.max_embed = max_embed
        self.delta_embed = self.max_embed - self.min_embed


class Cbox4CR(torch.nn.Module):
    def __init__(self, n_users, n_items, num_layers, gamma, epsilon, cen, activation,
                 beta, tau, std_vol,
                 emb_size=64, seed=2022):
        super(Cbox4CR, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_size = emb_size
        self.seed = seed
        self.num_layers = num_layers
        self.gamma = gamma
        self.epsilon = epsilon
        self.activation = activation
        self.cen = cen
        self.beta = beta  # crop/mask操作对原数据的增强部分占比
        self.tau = tau
        self.mask_default = self.mask_correlated_samples(batch_size=128)
        self.std_vol = std_vol
        self.box_type = 'normal'

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed((self.seed))
        random.seed(self.seed)

        self.gamma = nn.Parameter(
            torch.Tensor([self.gamma]),
            requires_grad=False
        )

        if self.activation == 'none':
            self.func = Identity
        elif self.activation == 'relu':
            self.func = F.relu
        elif self.activation == 'softplus':
            self.func = F.softplus

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / self.emb_size]),
            requires_grad=False
        )

        self.entity_embedding = nn.Parameter(torch.zeros(self.n_items + self.n_users, self.emb_size))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.offset_embedding = nn.Parameter(torch.zeros(3, self.emb_size))
        nn.init.uniform_(
            tensor=self.offset_embedding,
            a=0.,
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(3, self.emb_size))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        self.center_net = CenterIntersection(self.emb_size)
        self.offset_net = BoxOffsetIntersection(self.emb_size)
        self.contrastive_loss = nn.CrossEntropyLoss()
        self.RoPE = RotaryEmbedding(self.emb_size)
        self.trans_nn = MultilayerNN(self.emb_size, self.emb_size)

    def init_embedding(self, vocab_size, embed_dim, init_value):
        distribution = uniform.Uniform(init_value[0], init_value[1])
        embedding = distribution.sample((vocab_size, embed_dim))
        return embedding

    #
    def aug_func(self, history_item_embs, history_item_offset, rel_embs, rel_offsets, his_sz):
        tmp_his_embs = history_item_embs.clone()
        tmp_his_offset = history_item_offset.clone()
        tmp_rel_embs = rel_embs.clone()
        tmp_rel_offset = rel_offsets.clone()

        reorder_index = random.sample(range(his_sz), his_sz)
        aug_embs = tmp_his_embs[:, reorder_index, :] + tmp_rel_embs[:, reorder_index, :]
        aug_offset = tmp_his_offset[:, reorder_index, :] + self.func(tmp_rel_offset[:, reorder_index, :])
        aug_embs = self.RoPE.rotate_queries_or_keys(aug_embs)
        samples_list = []
        samples_offset_list = []
        for j in range(0, his_sz):
            samples_list.append(aug_embs[:, j])
            samples_offset_list.append(aug_offset[:, j])
        pos_center_embedding = self.center_net(torch.stack(samples_list))
        pos_center_offset = self.offset_net(torch.stack(samples_offset_list))
        return pos_center_embedding, pos_center_offset

    # ===随机打乱顺序-->只需要打乱positional embedding
    def aug_reorder(self, history_item_embs, history_item_offset, rel_embs, rel_offsets, his_sz):
        tmp_his_embs = history_item_embs.clone()
        tmp_his_offset = history_item_offset.clone()
        tmp_rel_embs = rel_embs.clone()
        tmp_rel_offset = rel_offsets.clone()

        reorder_index = random.sample(range(his_sz), his_sz)
        aug_embs = tmp_his_embs[:, reorder_index, :] + tmp_rel_embs[:, reorder_index, :]
        aug_offset = tmp_his_offset[:, reorder_index, :] + self.func(tmp_rel_offset[:, reorder_index, :])
        aug_embs = self.RoPE.rotate_queries_or_keys(aug_embs)
        samples_list = []
        samples_offset_list = []
        for j in range(0, his_sz):
            samples_list.append(aug_embs[:, j])
            samples_offset_list.append(aug_offset[:, j])
        pos_center_embedding = self.center_net(torch.stack(samples_list))
        pos_center_offset = self.offset_net(torch.stack(samples_offset_list))
        return pos_center_embedding, pos_center_offset

    # ===随机crop掉histories，和feedback
    def aug_crop1(self, history_item_embs, history_item_offset, rel_embs, rel_offsets, his_sz):
        tmp_his_embs = history_item_embs.clone()
        tmp_his_offset = history_item_offset.clone()
        tmp_rel_embs = rel_embs.clone()
        tmp_rel_offset = rel_offsets.clone()
        crop_length = math.floor(his_sz * self.beta)
        crop_index = random.sample(range(his_sz - 1), crop_length)
        retain_index = list(range(his_sz))
        retain_index = list(set(retain_index).difference(set(crop_index)))
        new_his_sz = len(retain_index)
        aug_embs = tmp_his_embs[:, retain_index, :] + tmp_rel_embs[:, retain_index, :]
        aug_offset = tmp_his_offset[:, retain_index, :] + self.func(tmp_rel_offset[:, retain_index, :])
        aug_embs = self.RoPE.rotate_queries_or_keys(aug_embs)
        samples_list = []
        samples_offset_list = []
        for j in range(0, new_his_sz):
            samples_list.append(aug_embs[:, j])
            samples_offset_list.append(aug_offset[:, j])
        pos_center_embedding = self.center_net(torch.stack(samples_list))
        pos_center_offset = self.offset_net(torch.stack(samples_offset_list))
        return pos_center_embedding, pos_center_offset

    def aug_crop_center(self, history_item_embs, history_item_offset, rel_embs, rel_offsets, his_sz):
        tmp_his_embs = history_item_embs.clone()
        tmp_his_offset = history_item_offset.clone()
        tmp_rel_embs = rel_embs.clone()
        tmp_rel_offset = rel_offsets.clone()
        crop_length = math.floor(his_sz * self.beta)
        crop_index = random.sample(range(his_sz - 1), crop_length)
        retain_index = list(range(his_sz))
        retain_index = list(set(retain_index).difference(set(crop_index)))
        new_his_sz = len(retain_index)
        aug_embs = tmp_his_embs[:, retain_index, :] + tmp_rel_embs[:, retain_index, :]
        aug_offset = tmp_his_offset + self.func(tmp_rel_offset)
        aug_embs = self.RoPE.rotate_queries_or_keys(aug_embs)
        samples_list = []
        samples_offset_list = []
        for j in range(0, new_his_sz):
            samples_list.append(aug_embs[:, j])
        pos_center_embedding = self.center_net(torch.stack(samples_list))

        for k in range(0, his_sz):
            samples_offset_list.append(aug_offset[:, j])
        pos_center_offset = self.offset_net(torch.stack(samples_offset_list))
        return pos_center_embedding, pos_center_offset

    def aug_BackOff(self, history_item_embs, history_item_offset, rel_embs, rel_offset, his_sz, item_ids):
        tmp_his_embs = history_item_embs.clone()
        tmp_his_offset = history_item_offset.clone()
        tmp_rel_embs = rel_embs.clone()
        tmp_rel_offset = rel_offset.clone()
        embsExpLast = tmp_his_embs[:, 0:-1, :] + tmp_rel_embs[:, 0:-1, :]
        offsetExpLast = tmp_his_offset[:, 0:-1, :] + self.func(tmp_rel_offset[:, 0:-1, :])
        embsExpLast = self.RoPE.rotate_queries_or_keys(embsExpLast)
        last_embedding_list = []
        last_offset_embedding_list = []
        for i in range(0, his_sz - 1):
            last_embedding_list.append(embsExpLast[:, i])
            last_offset_embedding_list.append(offsetExpLast[:, i])

        last_item_embs = self.center_net(torch.stack(last_embedding_list))
        last_item_offset = self.offset_net(torch.stack(last_offset_embedding_list))
        last_item_embs, last_item_offset = self.trans_nn(last_item_embs, last_item_offset)
        last_reasoning_ids = torch.zeros_like(item_ids)
        last_reasoning_ids += 2
        rel_reasoning_embs = torch.index_select(self.relation_embedding, dim=0, index=last_reasoning_ids[:])
        rel_reasoning_offset = torch.index_select(self.offset_embedding, dim=0, index=last_reasoning_ids[:])
        last_item_embs = last_item_embs + rel_reasoning_embs
        last_item_offset = last_item_offset + rel_reasoning_offset

        all_history_embs = torch.cat([tmp_his_embs[:, 0:-1, :], last_item_embs.unsqueeze(dim=1)], dim=1)
        all_history_offset = torch.cat([tmp_his_offset[:, 0:-1, :], last_item_offset.unsqueeze(dim=1)], dim=1)

        cl_embs = all_history_embs + rel_embs
        cl_offset = all_history_offset + rel_offset
        cl_embs = self.RoPE.rotate_queries_or_keys(cl_embs)
        cl_embedding_list = []
        cl_offset_embedding_list = []
        for i in range(0, his_sz):
            cl_embedding_list.append(cl_embs[:, i])
            cl_offset_embedding_list.append(cl_offset[:, i])
        cl_embedding = self.center_net(torch.stack(cl_embedding_list))
        cl_offset_embedding = self.offset_net(torch.stack(cl_offset_embedding_list))
        return cl_embedding, cl_offset_embedding

    # ====随机mask掉embedding和rel_ebedding，用p_value填充
    def aug_mask(self, history_item_embs, history_item_offset, rel_embs, rel_offsets, his_sz):
        tmp_his_embs = history_item_embs.clone()
        tmp_his_offset = history_item_offset.clone()
        tmp_rel_embs = rel_embs.clone()
        tmp_rel_offset = rel_offsets.clone()
        mask_length = math.floor(his_sz * self.beta)
        mask_index = random.sample(range(his_sz - 1), mask_length)
        tmp_rel_embs[:, mask_index, :] = self.cen_pad_value
        retain_index = list(range(his_sz))
        retain_index = list(set(retain_index).difference(set(mask_index)))

        aug_embs = tmp_his_embs + tmp_rel_embs
        aug_offset = tmp_his_offset[:, retain_index, :] + self.func(tmp_rel_offset[:, retain_index, :])
        aug_embs = self.RoPE.rotate_queries_or_keys(aug_embs)
        samples_list = []
        samples_offset_list = []
        for j in range(0, his_sz):
            samples_list.append(aug_embs[:, j])
            samples_offset_list.append(aug_offset[:, j])
        pos_center_embedding = self.center_net(torch.stack(samples_list))
        pos_center_offset = self.offset_net(torch.stack(samples_offset_list))
        return pos_center_embedding, pos_center_offset

    def cal_logit_box(self, entity_embedding, query_center_embedding, query_offset_embedding):
        delta = (entity_embedding - query_center_embedding).abs()
        distance_out = F.relu(delta - query_offset_embedding)
        distance_in = torch.min(delta, query_offset_embedding)
        logit = self.gamma - torch.norm(distance_out, p=1, dim=-1) - self.cen * torch.norm(distance_in, p=1, dim=-1)
        return logit

    def cal_logit_box4i2u(self, entity_embedding, query_center_embedding, query_offset_embedding):
        delta = (entity_embedding - query_center_embedding).abs()
        distance_out = F.relu(delta - query_offset_embedding)
        distance_in = torch.min(delta, query_offset_embedding)
        logit = self.gamma / 2 - torch.norm(distance_out, p=1, dim=-1) - self.cen * torch.norm(distance_in, p=1, dim=-1)
        return logit

    def positionalencoding1d(self, d_model, length, batchsz):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                              -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term) / 5
        pe[:, 1::2] = torch.cos(position.float() * div_term) / 5

        positionalE = pe.view(1, pe.size(0), pe.size(1))
        positionalE = positionalE.expand(batchsz, pe.size(0), pe.size(1)).to(device)

        return positionalE

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def infoNce(self, sim, batchsz):
        N = 2 * batchsz

        sim = sim / self.tau

        sim_i_j = torch.diag(sim, batchsz)
        sim_j_i = torch.diag(sim, -batchsz)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if batchsz != 128:
            mask = self.mask_correlated_samples(batchsz)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.contrastive_loss(logits, labels)
        return loss

    def log_inter_volumes(self, inter_min, inter_max, scale=1.):
        eps = 1e-16
        if isinstance(scale, float):
            s = torch.tensor(scale)
        else:
            s = scale
        log_vol = torch.sum(
            torch.log(
                F.softplus(inter_max - inter_min, beta=0.7).clamp_min(eps)
            ),
            dim=-1
        ) + torch.log(s)
        return log_vol

    def log_gumbel_volumes(self, inter_min, inter_max, scale=1.):
        eps = 1e-16
        delta = inter_max - inter_min
        if isinstance(scale, float):
            s = torch.tensor(scale)
        else:
            s = scale
        log_vol = torch.sum(
            torch.log(
                F.softplus(delta - 2 * self.euler_gamma * self.gumbel_beta, beta=1).clamp_min(eps)
            ),
            dim=-1
        ) + torch.log(s)
        return log_vol

    def intersection(self, box1_min, box1_max, box2_min, box2_max):
        intersections_min = self.gumbel_beta * torch.logsumexp(
            torch.stack((box1_min / self.gumbel_beta, box2_min / self.gumbel_beta)),
            0
        )
        intersections_min = torch.max(
            intersections_min,
            torch.max(box1_min, box2_min)
        )

        intersections_max = - self.gumbel_beta * torch.logsumexp(
            torch.stack((-box1_max / self.gumbel_beta, -box2_max / self.gumbel_beta)),
            0
        )
        intersections_max = torch.min(
            intersections_max,
            torch.min(box1_max, box2_max)
        )
        inter_box = Box(intersections_min, intersections_max)
        return inter_box

    def normal_intersection(self, boxes1, boxes2):

        intersections_min = torch.max(boxes1.min_embed, boxes2.min_embed)
        intersections_max = torch.min(boxes1.max_embed, boxes2.max_embed)

        inter_box = Box(intersections_min, intersections_max)
        return inter_box

    def outersection_length(self, boxes1, boxes2):
        outersections_min = torch.min(boxes1.min_embed, boxes2.min_embed)
        outersections_max = torch.max(boxes1.max_embed, boxes2.max_embed)
        c2 = torch.norm(outersections_max - outersections_min, p=2, dim=-1)
        return c2

    def diou_loss(self, box1_embs, box1_off, box2_embs, box2_off, box_type):
        box1_min = (box1_embs - F.relu(box1_off))
        box1_max = (box1_embs + F.relu(box1_off))
        box2_min = (box2_embs - F.relu(box2_off))
        box2_max = (box2_embs + F.relu(box2_off))
        if box_type == 'normal':
            inter_min = torch.max(box1_min, box2_min)
            inter_max = torch.min(box1_max, box2_max)
            cen_dis = torch.norm(box1_embs - box2_embs, p=2, dim=-1)
            outer_min = torch.min(box1_min, box2_min)
            outer_max = torch.max(box1_max, box2_max)
            inter_vol = self.log_inter_volumes(inter_min, inter_max)
        elif box_type == 'gumbel':
            inter_min = self.gumbel_beta * torch.logsumexp(
                torch.stack((box1_min / self.gumbel_beta, box2_min / self.gumbel_beta)),
                0
            )
            inter_min = torch.max(
                inter_min,
                torch.max(box1_min, box2_min)
            )

            inter_max = - self.gumbel_beta * torch.logsumexp(
                torch.stack((-box1_max / self.gumbel_beta, -box2_max / self.gumbel_beta)),
                0
            )
            inter_max = torch.min(
                inter_max,
                torch.min(box1_max, box2_max)
            )
            inter_vol = self.log_gumbel_volumes(inter_min, inter_max)
        c2 = torch.norm(outer_max - outer_min, p=2, dim=-1)
        d_loss = torch.sqrt(cen_dis) / torch.sqrt(c2)
        logit = torch.sigmoid(inter_vol+self.std_vol) - d_loss

        return logit

    def forward(self, batch_data):
        user_ids, item_ids, rating, histories, history_feedbacks, neg_item_ids, query_type, neg_user_ids = batch_data

        # [item]--0/1-->[user]  query_shape_1
        if query_type[0] == 1:
            neg_num = neg_user_ids.size(1)
            batchsz = item_ids.size(0)
            item_embs = torch.index_select(self.entity_embedding, dim=0, index=item_ids[:])
            item_offset = torch.zeros_like(item_embs).to(device)
            neg_user_embs = torch.index_select(self.entity_embedding, dim=0, index=neg_user_ids[:].view(-1)).view(
                batchsz,
                neg_num,
                self.emb_size)
            user_embs = torch.index_select(self.entity_embedding, dim=0, index=user_ids[:])
            rel_embs = torch.index_select(self.relation_embedding, dim=0, index=rating[:])
            rel_offset = torch.index_select(self.offset_embedding, dim=0, index=rating[:])

            # projection
            item_embs = item_embs
            item_offset = item_offset

            query_embs = item_embs + rel_embs
            query_offset = item_offset + self.func(rel_offset)

            # expand neg query
            query_embs4neg = query_embs.view(query_embs.size(0), 1, query_embs.size(1))
            query_embs4neg = query_embs4neg.expand(query_embs.size(0), neg_num, query_embs.size(1))

            query_offset4neg = query_offset.view(query_offset.size(0), 1, query_offset.size(1))
            query_offset4neg = query_offset4neg.expand(query_offset.size(0), neg_num, query_offset.size(1))

            # compute distance
            positive_logit = self.cal_logit_box4i2u(user_embs, query_embs, query_offset)
            negative_logit = self.cal_logit_box4i2u(neg_user_embs, query_embs4neg, query_offset4neg)
            loss = torch.zeros([])
            loss.to(device)
            return positive_logit, negative_logit, 0
        # [item1]-
        #         -
        #          -
        #           -
        # [item2]---->[user]--3-->[item*]    query_shape
        #           -
        #          -
        #         -
        # [item3]-
        if query_type[0] == 4:
            neg_num = neg_item_ids.size(1)
            batchsz = item_ids.size(0)
            his_sz = histories.size(1)
            item_embs = torch.index_select(self.entity_embedding, dim=0, index=item_ids[:])
            neg_item_embs = torch.index_select(self.entity_embedding, dim=0, index=neg_item_ids[:].view(-1)).view(
                batchsz,
                neg_num,
                self.emb_size)
            history_item_embs = torch.index_select(self.entity_embedding, dim=0, index=histories[:].view(-1)).view(
                batchsz,
                his_sz,
                self.emb_size)
            history_item_offset = torch.zeros_like(history_item_embs).to(device)
            rel_embs = torch.index_select(self.relation_embedding, dim=0, index=history_feedbacks[:].view(-1)).view(
                batchsz,
                his_sz,
                self.emb_size)
            rel_offset = torch.index_select(self.offset_embedding, dim=0, index=history_feedbacks[:].view(-1)).view(
                batchsz,
                his_sz,
                self.emb_size)

            # ==== 对比学习样本对 ====
            aug1_embs, aug1_offset = self.aug_crop1(history_item_embs, history_item_offset, rel_embs, rel_offset,
                                                    his_sz)
            aug1_embs, aug1_offset = self.trans_nn(aug1_embs, aug1_offset)

            history_item_embs = history_item_embs
            history_item_offset = history_item_offset

            embs = history_item_embs + rel_embs
            offset = history_item_offset + self.func(rel_offset)
            embs = self.RoPE.rotate_queries_or_keys(embs)

            embedding_list = []
            offset_embedding_list = []
            for i in range(0, his_sz):
                embedding_list.append(embs[:, i])
                offset_embedding_list.append(offset[:, i])
            embedding = self.center_net(torch.stack(embedding_list))
            offset_embedding = self.offset_net(torch.stack(offset_embedding_list))
            embedding, offset_embedding = self.trans_nn(embedding, offset_embedding)

            rel_buy_type = torch.zeros_like(item_ids)
            rel_buy_type += 2
            buy_embs = torch.index_select(self.relation_embedding, dim=0, index=rel_buy_type[:])
            buy_offset = torch.index_select(self.offset_embedding, dim=0, index=rel_buy_type[:])
            query_embs = embedding + buy_embs
            query_offset = offset_embedding + self.func(buy_offset)

            # 对比学习
            neg_sample_embs = torch.cat([embedding, aug1_embs], dim=0)
            neg_sample_embs1 = neg_sample_embs.view(1, neg_sample_embs.size(0), neg_sample_embs.size(1))
            neg_sample_embs1 = neg_sample_embs1.expand(batchsz * 2, neg_sample_embs1.size(1), neg_sample_embs1.size(2))
            neg_sample_offset = torch.cat([offset_embedding, aug1_offset], dim=0)
            neg_sample_offset1 = neg_sample_offset.view(1, neg_sample_offset.size(0), neg_sample_offset.size(1))
            neg_sample_offset1 = neg_sample_offset1.expand(batchsz * 2, neg_sample_offset1.size(1),
                                                           neg_sample_offset1.size(2))

            neg_sample_embs2 = neg_sample_embs.view(neg_sample_embs.size(0), 1, neg_sample_embs.size(1))
            neg_sample_embs2 = neg_sample_embs2.expand(neg_sample_embs2.size(0), batchsz * 2, neg_sample_embs2.size(2))
            neg_sample_offset2 = neg_sample_offset.view(neg_sample_offset.size(0), 1, neg_sample_offset.size(1))
            neg_sample_offset2 = neg_sample_offset2.expand(neg_sample_offset2.size(0), batchsz * 2,
                                                           neg_sample_offset2.size(2))

            logit = self.diou_loss(neg_sample_embs1, neg_sample_offset1, neg_sample_embs2, neg_sample_offset2, self.box_type)
            cl_loss = self.infoNce(logit, batchsz)

            query_embs4neg = query_embs.view(query_embs.size(0), 1, query_embs.size(1))
            query_embs4neg = query_embs4neg.expand(query_embs.size(0), neg_item_ids.size(1), query_embs.size(1))

            query_offset4neg = query_offset.view(query_offset.size(0), 1, query_offset.size(1))
            query_offset4neg = query_offset4neg.expand(query_offset.size(0), neg_item_ids.size(1), query_offset.size(1))

            positive_logit = self.cal_logit_box(item_embs, query_embs, query_offset)
            negative_logit = self.cal_logit_box(neg_item_embs, query_embs4neg, query_offset4neg)
            return positive_logit, negative_logit, cl_loss

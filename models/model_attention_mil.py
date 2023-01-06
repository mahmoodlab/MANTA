import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np

from collections import OrderedDict
from os.path import join
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class XlinearFusion(nn.Module):
    r"""
    Late Fusion Block using Bilinear Pooling
    args:
        skip (int): Whether to input features at the end of the layer
        use_bilinear (bool): Whether to use bilinear pooling during information gating
        gate1 (bool): Whether to apply gating to modality 1
        gate2 (bool): Whether to apply gating to modality 2
        dim1 (int): Feature mapping dimension for modality 1
        dim2 (int): Feature mapping dimension for modality 2
        scale_dim1 (int): Scalar value to reduce modality 1 before the linear layer
        scale_dim2 (int): Scalar value to reduce modality 2 before the linear layer
        mmhid (int): Feature mapping dimension after multimodal fusion
        dropout_rate (float): Dropout rate
    """
    def __init__(self, skip=1, use_bilinear=0, gate=1, dim=256, scale_dim=16, num_modalities=4,
                 mmhid1=256, mmhid2=256, dropout_rate=0.25):
        super(XlinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate = gate
        self.num_modalities = num_modalities

        dim_og, dim = dim, dim//scale_dim
        skip_dim = dim_og*self.num_modalities if skip else 0
                
        self.reduce = []
        for i in range(self.num_modalities):
            linear_h = nn.Sequential(nn.Linear(dim_og, dim), nn.ReLU())
            linear_z = nn.Bilinear(dim_og, dim_og, dim) if use_bilinear else nn.Sequential(nn.Linear(dim_og*self.num_modalities, dim))
            linear_o = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(p=dropout_rate))
            
            if self.gate:
                self.reduce.append(nn.ModuleList([linear_h, linear_z, linear_o]))
            else:
                self.reduce.append(nn.ModuleList([linear_h, linear_o]))
                
        self.reduce = nn.ModuleList(self.reduce)

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(nn.Linear((dim+1)**num_modalities, mmhid1), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.encoder2 = nn.Sequential(nn.Linear(mmhid1+skip_dim, mmhid2), nn.ReLU(), nn.Dropout(p=dropout_rate))

    def forward(self, v_list: list):
        v_cat = torch.cat(v_list, axis=1)
        o_list = []
        
        for i, v in enumerate(v_list):
            h = self.reduce[i][0](v)
            z = self.reduce[i][1](v_cat)
            o = self.reduce[i][2](nn.Sigmoid()(z)*h)
            o = torch.cat((o, torch.cuda.FloatTensor(o.shape[0], 1).fill_(1)), 1)
            o_list.append(o)
        
        o_fusion = o_list[0]
        for o in o_list[1:]:
            o_fusion = torch.bmm(o_fusion.unsqueeze(2), o.unsqueeze(1)).flatten(start_dim=1)

        ### Fusion
        out = self.post_fusion_dropout(o_fusion)
        out = self.encoder1(out)
        if self.skip: 
            for v in v_list:
                out = torch.cat((out, v), axis=1)
        out = self.encoder2(out)
        return out

"""
A Modified Implementation of Deep Attention MIL
"""


"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes (experimental usage for multiclass MIL)
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 768, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes (experimental usage for multiclass MIL)
"""
class Attn_Net_Gated(nn.Module):

    def __init__(self, L = 768, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


"""
Attention MIL with 1 additional hidden fc-layer after CNN feature extractor
args:
    gate: whether to use gating in attention network
    size_args: size config of attention network
    dropout: whether to use dropout in attention network
    n_classes: number of classes
"""

class MIL_Attention_fc_mtl(nn.Module):
    def __init__(self, gate = True, size_arg = "big", dropout = False, n_classes = 2):
        super(MIL_Attention_fc_mtl, self).__init__()
        self.size_dict = {"small": [768, 512, 256], "big": [768, 512, 384]}
        size = self.size_dict[size_arg]

        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        fc.extend([nn.Linear(size[1], size[1]), nn.ReLU()])
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 3)

        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 3)

        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifier_1 = nn.Linear(size[1], n_classes[0])
        self.classifier_2 = nn.Linear(size[1], n_classes[1])
        self.classifier_3 = nn.Linear(size[1], n_classes[2])

        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to('cuda:0')

        else:
            self.attention_net = self.attention_net.to(device)


        self.classifier_1 = self.classifier_1.to(device)
        self.classifier_2 = self.classifier_2.to(device)
        self.classifier_3 = self.classifier_3.to(device)

    def forward(self, h, return_features=False, attention_only=False):
        A, h = self.attention_net(h)
        A = torch.transpose(A, 1, 0)
        if attention_only:
            return A[0]

        A_raw = A
        A = F.softmax(A, dim=1)
        M = torch.mm(A, h)

        logits_task1  = self.classifier_1(M[0].unsqueeze(0))
        Y_hat_task1   = torch.topk(logits_task1, 1, dim = 1)[1]
        Y_prob_task1  = F.softmax(logits_task1, dim = 1)

        logits_task2  = self.classifier_2(M[1].unsqueeze(0))
        Y_hat_task2   = torch.topk(logits_task2, 1, dim = 1)[1]
        Y_prob_task2  = F.softmax(logits_task2, dim = 1)

        logits_task3  = self.classifier_3(M[2].unsqueeze(0))
        Y_hat_task3   = torch.topk(logits_task3, 1, dim = 1)[1]
        Y_prob_task3  = F.softmax(logits_task3, dim = 1)

        results_dict = {}
        if return_features:
            results_dict.update({'features': M})

        results_dict.update({'logits_task1': logits_task1, 'Y_prob_task1': Y_prob_task1, 'Y_hat_task1': Y_hat_task1,
                             'logits_task2': logits_task2, 'Y_prob_task2': Y_prob_task2, 'Y_hat_task2': Y_hat_task2,
                             'logits_task3': logits_task3, 'Y_prob_task3': Y_prob_task3, 'Y_hat_task3': Y_hat_task3,
                             'A': A_raw})

        return results_dict


"""
Attention MIL (with multiple stains) with 1 additional hidden fc-layer after CNN feature extractor
args:
    num_stains: number of stains
    gate: whether to use gating in attention network
    size_args: size config of attention network
    dropout: whether to use dropout in attention network
    n_classes: number of classes
"""
class MIL_Attention_fc_mtl_ms(nn.Module):
    def __init__(self, stains = ['HnE', 'Jon', 'Tri', 'PAS'], fusion='concat', gate = True, size_arg = "small", 
                 dropout = 0.25, n_classes = [2,2,3]):
        super(MIL_Attention_fc_mtl_ms, self).__init__()
        self.stains = stains
        self.size_dict = {"small": [768, 256, 256], "big": [768, 512, 384]}
        size = self.size_dict[size_arg]
        self.fusion = fusion
        
        if fusion == 'max':
            ms_fc_net = []
            for _ in range(len(stains)):
                fc = [nn.Dropout(0.10), nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(0.25)]
                ms_fc_net.append(nn.Sequential(*fc))
            self.ms_fc_net = nn.ModuleList(ms_fc_net)
            self.global_attn_net = Attn_Net_Gated(L=size[1], D=size[1], dropout=0.25, n_classes=3)
            self.classifier_1 = nn.Linear(256, n_classes[0])
            self.classifier_2 = nn.Linear(256, n_classes[1])
            self.classifier_3 = nn.Linear(256, n_classes[2])
        else:
            ms_attention_net = []
            for _ in range(len(stains)):
                fc = [nn.Dropout(0.10), nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(0.25)]
                fc.append(Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 3))
                ms_attention_net.append(nn.Sequential(*fc))
            self.ms_attention_net = nn.ModuleList(ms_attention_net)

            if fusion=='concat':
                self.classifier_1 = nn.Linear(size[1]*len(self.stains), n_classes[0])
                self.classifier_2 = nn.Linear(size[1]*len(self.stains), n_classes[1])
                self.classifier_3 = nn.Linear(size[1]*len(self.stains), n_classes[2])
            elif fusion=='tensor':
                self.xfusion_1 = XlinearFusion(dim=256, scale_dim=16)
                self.xfusion_2 = XlinearFusion(dim=256, scale_dim=16)
                self.xfusion_3 = XlinearFusion(dim=256, scale_dim=16)
                self.classifier_1 = nn.Linear(256, n_classes[0])
                self.classifier_2 = nn.Linear(256, n_classes[1])
                self.classifier_3 = nn.Linear(256, n_classes[2])
            elif fusion == 'hierarchical_t':
                task1_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=0.25, activation='relu')
                task2_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=0.25, activation='relu')
                task3_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=0.25, activation='relu')
                self.task1_transformer = nn.TransformerEncoder(task1_encoder_layer, num_layers=1)
                self.task2_transformer = nn.TransformerEncoder(task2_encoder_layer, num_layers=1)
                self.task3_transformer = nn.TransformerEncoder(task3_encoder_layer, num_layers=1)
                self.task1_pooling = Attn_Net_Gated(L = 256, D = 256, dropout = 0.25, n_classes = 1)
                self.task2_pooling = Attn_Net_Gated(L = 256, D = 256, dropout = 0.25, n_classes = 1)
                self.task3_pooling = Attn_Net_Gated(L = 256, D = 256, dropout = 0.25, n_classes = 1)
                self.classifier_1 = nn.Linear(256, n_classes[0])
                self.classifier_2 = nn.Linear(256, n_classes[1])
                self.classifier_3 = nn.Linear(256, n_classes[2])


    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.fusion == 'max':
            self.ms_fc_net = self.ms_fc_net.to(device)
            self.global_attn_net = self.global_attn_net.to(device)
            
        else:
            self.ms_attention_net = self.ms_attention_net.to(device)
            if self.fusion == 'tensor':
                self.xfusion_1 = self.xfusion_1.to(device)
                self.xfusion_2 = self.xfusion_2.to(device)
                self.xfusion_3 = self.xfusion_3.to(device)
            elif self.fusion == 'hierarchical_t':
                self.task1_transformer = self.task1_transformer.to(device)
                self.task2_transformer = self.task2_transformer.to(device)
                self.task3_transformer = self.task3_transformer.to(device)
                self.task1_pooling = self.task1_pooling.to(device)
                self.task2_pooling = self.task2_pooling.to(device)
                self.task3_pooling = self.task3_pooling.to(device)

        self.classifier_1 = self.classifier_1.to(device)
        self.classifier_2 = self.classifier_2.to(device)
        self.classifier_3 = self.classifier_3.to(device)

    def forward(self, **kwargs):
        
        if self.fusion == 'max':
            M_all = []
            for idx, stain in enumerate(self.stains):
                h = self.ms_fc_net[idx](kwargs[stain])
                M_all.append(h)
                
            M_all = torch.cat(M_all, axis=0)
            A_all, M_all = self.global_attn_net(M_all)
            M_all = torch.mm(F.softmax(torch.transpose(A_all, 1, 0), dim=1), M_all)
        else:
            M_all = []
            A_all = {'task':None, 'stain':{}}
            for idx, stain in enumerate(self.stains):
                A, h = self.ms_attention_net[idx](kwargs[stain])
                A = torch.transpose(A, 1, 0)
                A_raw = A
                A_all['stain'][stain] = A.detach().cpu().numpy()
                A = F.softmax(A, dim=1)
                M = torch.mm(A, h)
                M_all.append(M)
                
            if self.fusion=='concat':
                M_all = torch.cat(M_all, axis=1)
                
            elif self.fusion=='tensor':
                M_1 = self.xfusion_1(v_list=[M[0].unsqueeze(dim=0) for M in M_all])
                M_2 = self.xfusion_2(v_list=[M[1].unsqueeze(dim=0) for M in M_all])
                M_3 = self.xfusion_3(v_list=[M[2].unsqueeze(dim=0) for M in M_all])
                M_all = torch.cat([M_1, M_2, M_3], axis=0)
                
            elif self.fusion == 'hierarchical':
                M_all = torch.cat(M_all, axis=1)
                h_task1 = M_all[0].reshape(4, 256)
                h_task2 = M_all[1].reshape(4, 256)
                h_task3 = M_all[2].reshape(4, 256)
                A_1, M_1 = self.task1_pooling(h_task1)
                A_2, M_2 = self.task2_pooling(h_task2)
                A_3, M_3 = self.task3_pooling(h_task3)
                M_1 = torch.mm(F.softmax(torch.transpose(A_1, 1, 0), dim=1), M_1)
                M_2 = torch.mm(F.softmax(torch.transpose(A_2, 1, 0), dim=1), M_2)
                M_3 = torch.mm(F.softmax(torch.transpose(A_3, 1, 0), dim=1), M_3)
                M_all = torch.cat([M_1, M_2, M_3], axis=0)
                A_all['task'] = torch.stack([A_1, A_2, A_3], axis=0).detach().cpu().numpy()
                
            elif self.fusion == 'hierarchical_t':
                M_all = torch.cat(M_all, axis=1)
                h_task1 = M_all[0].reshape(4, 256).unsqueeze(axis=1)
                h_task2 = M_all[1].reshape(4, 256).unsqueeze(axis=1)
                h_task3 = M_all[2].reshape(4, 256).unsqueeze(axis=1)
                h_task1 = self.task1_transformer(h_task1).squeeze(axis=1)
                h_task2 = self.task2_transformer(h_task2).squeeze(axis=1)
                h_task3 = self.task3_transformer(h_task3).squeeze(axis=1)
                A_1, M_1 = self.task1_pooling(h_task1)
                A_2, M_2 = self.task2_pooling(h_task2)
                A_3, M_3 = self.task3_pooling(h_task3)
                M_1 = torch.mm(F.softmax(torch.transpose(A_1, 1, 0), dim=1), M_1)
                M_2 = torch.mm(F.softmax(torch.transpose(A_2, 1, 0), dim=1), M_2)
                M_3 = torch.mm(F.softmax(torch.transpose(A_3, 1, 0), dim=1), M_3)
                M_all = torch.cat([M_1, M_2, M_3], axis=0)
                A_all['task'] = torch.stack([A_1, A_2, A_3], axis=0).detach().cpu().squeeze(dim=2).numpy()
            
        logits_task1  = self.classifier_1(M_all[0].unsqueeze(0))
        Y_hat_task1   = torch.topk(logits_task1, 1, dim = 1)[1]
        Y_prob_task1  = F.softmax(logits_task1, dim = 1)

        logits_task2  = self.classifier_2(M_all[1].unsqueeze(0))
        Y_hat_task2   = torch.topk(logits_task2, 1, dim = 1)[1]
        Y_prob_task2  = F.softmax(logits_task2, dim = 1)

        logits_task3  = self.classifier_3(M_all[2].unsqueeze(0))
        Y_hat_task3   = torch.topk(logits_task3, 1, dim = 1)[1]
        Y_prob_task3  = F.softmax(logits_task3, dim = 1)

        results_dict = {}
        if kwargs['return_features']:
            results_dict.update({'features': M})

        results_dict.update({'logits_task1': logits_task1, 'Y_prob_task1': Y_prob_task1, 'Y_hat_task1': Y_hat_task1,
                             'logits_task2': logits_task2, 'Y_prob_task2': Y_prob_task2, 'Y_hat_task2': Y_hat_task2,
                             'logits_task3': logits_task3, 'Y_prob_task3': Y_prob_task3, 'Y_hat_task3': Y_hat_task3,
                             'A': A_all})
        return results_dict

    def captum_forward(self, HnE, Jon, Tri, PAS, task):
        kwargs = {'HnE': HnE,
                  'Jon': Jon,
                  'Tri': Tri,
                  'PAS': PAS}

        ### 1. Processing Each WSI stain via stain-specific Attention MIL Pooling
        M_all = []
        for idx, stain in enumerate(self.stains):
            A, h = self.ms_attention_net[idx](kwargs[stain])
            A = A.squeeze(dim=2)
            A = F.softmax(A, dim=1) # These attention weights correspond with raw patch importance within each stain
            M = torch.bmm(torch.transpose(A, 1, 2), h)
            M_all.append(M)
            
        ### 2. Concat of Stain Embeddings (along Task Dimension)
        M_all = torch.cat(M_all, axis=2)

        ### 3. Reshaping Task Feature Embeddings
        h_task1 = M_all[:,0,:].reshape(-1, 4, 256)
        h_task2 = M_all[:,1,:].reshape(-1, 4, 256)
        h_task3 = M_all[:,2,:].reshape(-1, 4, 256)
        
        if self.fusion == 'hierarchical_t':
            h_task1 = self.task1_transformer(h_task1)
            h_task2 = self.task2_transformer(h_task2)
            h_task3 = self.task3_transformer(h_task3)

        ### 4. Getting Stain Attention Weights For Each Task
        A_1, M_1 = self.task1_pooling(h_task1) 
        A_2, M_2 = self.task2_pooling(h_task2)
        A_3, M_3 = self.task3_pooling(h_task3)

        ### 5. Pooling Each WSI Stain For Each Task
        M_1 = torch.bmm(F.softmax(A_1.squeeze(dim=2), dim=1).unsqueeze(dim=1), M_1).squeeze(dim=1)
        M_2 = torch.bmm(F.softmax(A_2.squeeze(dim=2), dim=1).unsqueeze(dim=1), M_2).squeeze(dim=1)
        M_3 = torch.bmm(F.softmax(A_3.squeeze(dim=2), dim=1).unsqueeze(dim=1), M_3).squeeze(dim=1)

        
        ### 6. Multi-Task
        if task == 1:
            logits_task1  = self.classifier_1(M_1)
            Y_hat_task1   = torch.topk(logits_task1, 1, dim = 1)[1]
            Y_prob_task1  = F.softmax(logits_task1, dim = 1)
            return Y_prob_task1
        elif task == 2:
            logits_task2  = self.classifier_2(M_2)
            Y_hat_task2   = torch.topk(logits_task2, 1, dim = 1)[1]
            Y_prob_task2  = F.softmax(logits_task2, dim = 1)
            return Y_prob_task2
        elif task == 3:
            logits_task3  = self.classifier_3(M_3)
            Y_hat_task3   = torch.topk(logits_task3, 1, dim = 1)[1]
            Y_prob_task3  = F.softmax(logits_task3, dim = 1)
            return Y_prob_task3

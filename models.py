import torch
import torch.nn as nn
import torch.nn.functional as F


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)  # 1 x num_patches x 256
        b = self.attention_b(x)  # 1 x num_patches x 256
        A = a.mul(b)  # 1 x num_patches x 256
        A = self.attention_c(A)  # N x n_tasks, num_patches x 512
        return A, x


"""
/data/zhongz2/results_histo256_generated7fp_hf_TCGA-ALL2_32_2gpus/adam_RegNormNone_Encoderimagenetmobilenetv3_CLSLOSSweighted_ce_accum4_wd1e-4_reguNone1e-4/split_3/snapshot_22.pt
"""

# survival not shared, all other shared
class AttentionModel_bak(nn.Module):
    def __init__(self):
        super().__init__()
        fc = [nn.Linear(1280, 256)]
        self.attention_net = nn.Sequential(*fc)

        self.initialize_weights()

    def initialize_weights(self):
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, label=None, instance_eval=False, return_features=False, attention_only=False):

        x_path = x.squeeze(0)  # 1 x num_patches x 512  1 x 10000 x 512 --> all 256x256 pacthces

        return self.attention_net(x_path)



# survival not shared, all other shared
class AttentionModel(nn.Module):
    def __init__(self, classification_dict, regression_list, args):
        super().__init__()

        self.has_cls = len(classification_dict) > 0
        self.has_reg = len(regression_list) > 0
        self.args = args
        self.classification_dict = classification_dict
        self.regression_list = regression_list

        if args.attention_arch == 'shared_attention':
            self.num_tasks = 1
            self.index_step = 0
        elif args.attention_arch == 'multiple_attention':
            self.num_tasks = len(self.classification_dict)
            self.num_tasks += len(self.regression_list)
            self.index_step = 1

        if 'patch' in self.args.model_name:
            self.feature_extractor = tv_resnet_ae.__dict__[args.backbone](pretrained=True)
            latent_dim = self.feature_extractor.latent_dim
            # for param in self.feature_extractor.parameters():   # fix the feature extraction network
            #     param.requires_grad = False

            L = latent_dim  # 512
            D = 256  # latent_dim
            K = self.num_tasks
            a1 = [nn.Linear(L, D), nn.Tanh()]
            a2 = [nn.Linear(L, D), nn.Sigmoid()]
            if 0 < args.dropout < 1:
                a1.append(nn.Dropout(args.dropout))
                a2.append(nn.Dropout(args.dropout))
            self.attention_V = torch.nn.Sequential(*a1)
            self.attention_U = torch.nn.Sequential(*a2)
            self.attention_weights = torch.nn.Linear(D, K)
        elif 'pretrain' in self.args.model_name:
            network_dims = {
                'swinv2': 768,
                'mobilevit': 640,
                'convnext': 768,
                'vit': 768,
                'beit': 768,
                'mobilenetv1': 1024,
                'resnet18': 512,
                'resnet50': 2048,
                'vithybrid': 768,
                'mobilenetv2': 1280,
                'mobilenetv3': 1280,
                'CLIP': 512,
                'PLIP': 512,
            }
            # size = [network_dims[args.backbone], network_dims[args.backbone]//2, network_dims[args.backbone]//2]
            size = [network_dims[args.backbone], 256, 256]
            # size = [512, 384, 256]  # feature after ResNet-18 is 512d
            fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(args.dropout)]
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=args.dropout, n_classes=self.num_tasks)
            fc.append(attention_net)
            self.attention_net = nn.Sequential(*fc)
            self.rho = nn.Sequential(*[nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(args.dropout)])
            L = size[2]
        else:
            raise ValueError('unsupported model_name')

        classifiers = {}
        for k, labels in self.classification_dict.items():
            classifiers[k] = nn.Linear(L, len(labels))
        self.classifiers = nn.ModuleDict(classifiers)
        regressors = {}
        for k in self.regression_list:
            regressors[k] = nn.Linear(L, 1)
        self.regressors = nn.ModuleDict(regressors)

        self.initialize_weights()

        if 'patch' in self.args.model_name:
            state_dict = load_state_dict_from_url(model_urls[args.backbone], progress=False)
            self.feature_extractor.load_state_dict(state_dict, strict=False)

        if args.fixed_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    def initialize_weights(self):
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, label=None, instance_eval=False, return_features=False, attention_only=False):
        if 'patch' in self.args.model_name:
            return self.forward_patch(x)
        elif 'pretrain' in self.args.model_name:
            return self.forward_pretrain(x, attention_only=attention_only)

    def forward_pretrain(self, x, attention_only=False):

        x_path = x.squeeze(0)  # 1 x num_patches x 512  1 x 10000 x 512 --> all 256x256 pacthces

        A, h = self.attention_net(x_path)  # num_patches x num_tasks, num_patches x 512
        A = torch.transpose(A, 1, 0)  # num_tasks x num_patches
        # A_raw = A  # 1 x num_patches
        if attention_only:
            return {'A_raw': A}

        results_dict = {}
        A = F.softmax(A, dim=1)  # num_tasks x num_patches, normalized
        h = torch.mm(A, h)  # A: num_tasks x num_patches, h_path: num_patches x 256  --> num_tasks x 256
        results_dict['global_feat'] = h
        results_dict['A'] = A
        # for task 1, exp(x_i) / \sum_i=1^{num_pathces} exp(x_i) --> [0, 1]
        h = self.rho(h)

        index = 0
        # results_dict = {'A_raw': A_raw}
        for k, classifier in self.classifiers.items():
            logits_k = classifier(h[index].unsqueeze(0))
            results_dict[k + '_logits'] = logits_k
            index += self.index_step

        for k, regressor in self.regressors.items():
            values_k = regressor(h[index].unsqueeze(0)).squeeze(1)
            results_dict[k + '_logits'] = values_k
            index += self.index_step

        return results_dict

    def forward_patch(self, x):

        # x: (BS x num_patches, num_channels, h, w)
        batch_size, num_patches, num_channels, h, w = x.size()
        x = torch.reshape(x, (batch_size * num_patches, num_channels, h, w))
        feat = self.feature_extractor(x)  # features after ResNet-18 512

        # for each slide, 10000 patches with size of 128x128
        # 2 x 10000 x 512 features
        # for other tasks
        feat = torch.reshape(feat, (batch_size, num_patches, -1))  # B x P x 512
        A_V = self.attention_V(feat)  # A_V: B x P x D
        A_U = self.attention_U(feat)  # A_U: B x P x D
        A = self.attention_weights(A_V * A_U)  # A: B x P x num_tasks
        A = F.softmax(A, dim=1)  # B x P x num_tasks
        h = torch.matmul(torch.transpose(A, 1, 2), feat)  # B x P x L  B x P x 256

        index = 0
        results_dict = {}
        for k, classifier in self.classifiers.items():
            logits_k = classifier(h[:, index, :])
            results_dict[k + '_logits'] = logits_k
            index += self.index_step

        for k, regressor in self.regressors.items():
            values_k = regressor(h[:, index, :]).squeeze(1)
            results_dict[k + '_logits'] = values_k
            index += self.index_step

        return results_dict





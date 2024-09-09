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


BACKBONE_DICT = {
    'CLIP': 512,
    'PLIP': 512,
    'MobileNetV3': 1280,
    'mobilenetv3': 1280,
    'ProvGigaPath': 1536,
    'CONCH': 512,
    'UNI': 1024
}

# survival not shared, all other shared
class AttentionModel(nn.Module):
    def __init__(self, backbone='PLIP'):
        super().__init__()

        self.classification_dict = {
            'CDH1_cls': ['Gain_Or_Unknown_Or_NaN', 'Loss', 'Other'],
            'GATA3_cls': ['Gain_Or_Unknown_Or_NaN', 'Loss', 'Other'],
            'PIK3CA_cls': ['Loss_Or_Unknown_Or_NaN', 'Gain', 'Other'],
            'TP53_cls': ['Gain_Or_Unknown_Or_NaN', 'Loss', 'Other'],
            'KRAS_cls': ['Loss_Or_Unknown_Or_NaN', 'Gain', 'Other'],
            'ARID1A_cls': ['Gain_Or_Unknown_Or_NaN', 'Loss', 'Other'],
            'PTEN_cls': ['Gain_Or_Unknown_Or_NaN', 'Loss', 'Other'],
            'BRAF_cls': ['Loss_Or_Unknown_Or_NaN', 'Gain', 'Other'],
            'APC_cls': ['Gain_Or_Unknown_Or_NaN', 'Loss', 'Other'],
            'ATRX_cls': ['Gain_Or_Unknown_Or_NaN', 'Loss', 'Other'],
            'IDH1_cls': ['Gain_Or_Unknown_Or_NaN', 'Loss_Or_Switch', 'Other']
        }
        self.regression_list = [
            'Cytotoxic_T_Lymphocyte',
            'TIDE_CAF',
            'TIDE_Dys',
            'TIDE_M2',
            'TIDE_MDSC',
            'HALLMARK_ADIPOGENESIS_sum',
            'HALLMARK_ALLOGRAFT_REJECTION_sum',
            'HALLMARK_ANDROGEN_RESPONSE_sum',
            'HALLMARK_ANGIOGENESIS_sum',
            'HALLMARK_APICAL_JUNCTION_sum',
            'HALLMARK_APICAL_SURFACE_sum',
            'HALLMARK_APOPTOSIS_sum',
            'HALLMARK_BILE_ACID_METABOLISM_sum',
            'HALLMARK_CHOLESTEROL_HOMEOSTASIS_sum',
            'HALLMARK_COAGULATION_sum',
            'HALLMARK_COMPLEMENT_sum',
            'HALLMARK_DNA_REPAIR_sum',
            'HALLMARK_E2F_TARGETS_sum',
            'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION_sum',
            'HALLMARK_ESTROGEN_RESPONSE_EARLY_sum',
            'HALLMARK_ESTROGEN_RESPONSE_LATE_sum',
            'HALLMARK_FATTY_ACID_METABOLISM_sum',
            'HALLMARK_G2M_CHECKPOINT_sum',
            'HALLMARK_GLYCOLYSIS_sum',
            'HALLMARK_HEDGEHOG_SIGNALING_sum',
            'HALLMARK_HEME_METABOLISM_sum',
            'HALLMARK_HYPOXIA_sum',
            'HALLMARK_IL2_STAT5_SIGNALING_sum',
            'HALLMARK_IL6_JAK_STAT3_SIGNALING_sum',
            'HALLMARK_INFLAMMATORY_RESPONSE_sum',
            'HALLMARK_INTERFERON_ALPHA_RESPONSE_sum',
            'HALLMARK_INTERFERON_GAMMA_RESPONSE_sum',
            'HALLMARK_KRAS_SIGNALING_DN_sum',
            'HALLMARK_KRAS_SIGNALING_UP_sum',
            'HALLMARK_MITOTIC_SPINDLE_sum',
            'HALLMARK_MTORC1_SIGNALING_sum',
            'HALLMARK_MYC_TARGETS_V1_sum',
            'HALLMARK_MYC_TARGETS_V2_sum',
            'HALLMARK_MYOGENESIS_sum',
            'HALLMARK_NOTCH_SIGNALING_sum',
            'HALLMARK_OXIDATIVE_PHOSPHORYLATION_sum',
            'HALLMARK_P53_PATHWAY_sum',
            'HALLMARK_PANCREAS_BETA_CELLS_sum',
            'HALLMARK_PEROXISOME_sum',
            'HALLMARK_PI3K_AKT_MTOR_SIGNALING_sum',
            'HALLMARK_PROTEIN_SECRETION_sum',
            'HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY_sum',
            'HALLMARK_SPERMATOGENESIS_sum',
            'HALLMARK_TGF_BETA_SIGNALING_sum',
            'HALLMARK_TNFA_SIGNALING_VIA_NFKB_sum',
            'HALLMARK_UNFOLDED_PROTEIN_RESPONSE_sum',
            'HALLMARK_UV_RESPONSE_DN_sum',
            'HALLMARK_UV_RESPONSE_UP_sum',
            'HALLMARK_WNT_BETA_CATENIN_SIGNALING_sum',
            'HALLMARK_XENOBIOTIC_METABOLISM_sum'
        ]

        self.attention_net = nn.Sequential(*[
            nn.Linear(BACKBONE_DICT[backbone], 256), 
            nn.ReLU(), 
            nn.Dropout(0.25),
            Attn_Net_Gated(L=256, D=256, dropout=0.25, n_classes=1)
        ])
        self.rho = nn.Sequential(*[nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.25)])

        classifiers = {}
        for k, labels in self.classification_dict.items():
            classifiers[k] = nn.Linear(256, len(labels))
        self.classifiers = nn.ModuleDict(classifiers)
        regressors = {}
        for k in self.regression_list:
            regressors[k] = nn.Linear(256, 1)
        self.regressors = nn.ModuleDict(regressors)

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
        h = self.rho(h)

        for k, classifier in self.classifiers.items():
            logits_k = classifier(h[0].unsqueeze(0))
            results_dict[k + '_logits'] = logits_k

        for k, regressor in self.regressors.items():
            values_k = regressor(h[0].unsqueeze(0)).squeeze(1)
            results_dict[k + '_logits'] = values_k

        return results_dict




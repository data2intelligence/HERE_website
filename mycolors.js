const CSS_COLOR_NAMES = [
    "Tomato",
    "Orange",
    "DodgerBlue",
    "Sienna",
    "Gray",
    "SlateBlue",
    "Violet",
    "LightGray",
    "Yellow",
    "Fuchsia",
    "Green",
    "Cyan",
    "Chartreuse",
    "OrangeRed",
    "Olive",
    "Teal",
    "Cornsilk",
    "Indigo",
    "Lightpink",
    "Lightsteelblue",
    "Skyblue",
    "Powderblue",
    "Limegreen",
    "Darkred"
];

const ALL_LABELS = {
    'PAM50_and_Claudin-low_(CLOW)_Molecular_Subtype':
        {'LumA': 0, 'LumB': 1, 'Basal': 2, 'HER2E':3, 'normal':4},
    'CLS_HistoAnno':
        {'Lobular':0, 'Ductal':1, 'Other':2},
    'CLS_IHC_HER2':
        {'Negative':0,  'Positive':1, 'Other':2},

    'survival_groups':
        {'LowRisk':0, 'HighRisk':1},
    'Stage':
        {'StageI_II':0, 'StageM':1},

    'treatment_group':
        {'1 (control Ab)':1, '2 (Anti-TGFb)':2},
    'met_weight_label':
        {'group0':0, 'group1':1, 'group2':2},

    'TP53_cls':
        {'Gain_Or_Unknown_Or_NaN':0, 'Loss':1, 'other':2},

    'ResponseGroup':
        {'PD':0, 'SD':1, 'PR':2, 'CR':3},
    'response_groups':
        {'BadResponse': 0, 'GoodResponse': 1},
    'GeneGroup':
        {'Vector': 0, 'SERPING1': 1},
    'group_label':
        {'group0': 0, 'group1': 1},
    'one_class':
        {'class0': 0},
    'one_patient':
        {'class0': 0},
    'FEC-T+Trastuzumab_group':
        {'BadResponse': 0, 'GoodResponse': 1},
    'T-FEC_group':
        {'BadResponse': 0, 'GoodResponse': 1},
    'ResistantSensitive_group':
        {'BadResponse': 0, 'GoodResponse': 1},
    'LumA_survival_group':
        {'LowRisk':0, 'HighRisk':1},
    'LumB_survival_group':
        {'LowRisk':0, 'HighRisk':1},
    'Basal_survival_group':
        {'LowRisk':0, 'HighRisk':1},
    'claudin-low_survival_group':
        {'LowRisk':0, 'HighRisk':1},
    'Her2_survival_group':
        {'LowRisk':0, 'HighRisk':1},
    'Normal_survival_group':
        {'LowRisk':0, 'HighRisk':1},
    'TriNeg_survival_group':
        {'LowRisk':0, 'HighRisk':1},
    'ControlAb_weight_label':
        {'group0':0, 'group1':1, 'group2':2},
    'AntiTGFb_weight_label':
        {'group0':0, 'group1':1, 'group2':2},
    '18C_SCC_response_group':
        {'BadResponse': 0, 'GoodResponse': 1},
    '18C_cervical_response_group':
        {'BadResponse': 0, 'GoodResponse': 1},
    '18C_HNSCC_response_group':
        {'BadResponse': 0, 'GoodResponse': 1},
    'HNSCC_response_group':
        {'BadResponse': 0, 'GoodResponse': 1},
    'Pre_response_group':
        {'BadResponse': 0, 'GoodResponse': 1},
    'ENSG00000149131.16_rna':
        {'LowGroup': 0, 'HighGroup': 1},
    'ENSG00000149131.16_proteome':
        {'LowGroup': 0, 'HighGroup': 1},
    'ENSG00000173372.17_rna':
        {'LowGroup': 0, 'HighGroup': 1},
    'ENSG00000173372.17_proteome':
        {'LowGroup': 0, 'HighGroup': 1},
    'ENSG00000136250.12_rna':
        {'LowGroup': 0, 'HighGroup': 1},
    'ENSG00000136250.12_proteome':
        {'LowGroup': 0, 'HighGroup': 1},
    'ENSG00000159403.18_rna':
        {'LowGroup': 0, 'HighGroup': 1},
    'ENSG00000159403.18_proteome':
        {'LowGroup': 0, 'HighGroup': 1}
};

import sys,os,json,glob
import pandas as pd
import numpy as np
import tarfile
import io
import gc
import re
import faiss
from sklearn.metrics import pairwise_distances
import base64
import lmdb
import torch
import torch.nn.functional as F
from models import AttentionModel
import torch.nn as nn
from torchvision import transforms
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
import timm
import pickle
import time
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.middleware.proxy_fix import ProxyFix

from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.use('agg')
import random
import pymysql
import pyarrow.parquet as pq
from transformers import CLIPModel, CLIPProcessor
import psutil
print(psutil.virtual_memory().used/1024/1024/1024, "GB")

def get_db():
    db_conn = pymysql.connect(
        user=os.environ['HERE_DB_USER'], 
        password=os.environ['HERE_DB_PASSWORD'], 
        host=os.environ['HERE_DB_HOST'], 
        database=os.environ['HERE_DB_DATABASE']
    )
    db_conn.autocommit = False
    db_cursor = db_conn.cursor()
    return db_conn, db_cursor

DATA_DIR = f'data_HiDARE_PLIP_20240208'
ST_ROOT = f'{DATA_DIR}/assets/ST_kmeans_clustering/analysis/one_patient_top_128'
project_names = ['TCGA-COMBINED', 'KenData_20240814', 'ST', 'CPTAC']  # do not change the order
project_start_ids = {'TCGA-COMBINED': 0, 'KenData_20240814': 159011314, 'ST': 281115587, 'CPTAC': 281293653}
backbones = ['HERE_CONCH'] # choices: 'HERE_CONCH', 'HERE_PLIP', 'HERE_ProvGigaPath', 'HERE_UNI'

def load_cfg_from_json(json_file):
    with open(json_file, "r", encoding="utf-8") as reader:
        text = reader.read()
    return json.loads(text)

def load_model_config_from_hf(model_id: str):
    cached_file = f'{DATA_DIR}/assets/ProvGigaPath/config.json'

    hf_config = load_cfg_from_json(cached_file)
    if 'pretrained_cfg' not in hf_config:
        # old form, pull pretrain_cfg out of the base dict
        pretrained_cfg = hf_config
        hf_config = {}
        hf_config['architecture'] = pretrained_cfg.pop('architecture')
        hf_config['num_features'] = pretrained_cfg.pop('num_features', None)
        if 'labels' in pretrained_cfg:  # deprecated name for 'label_names'
            pretrained_cfg['label_names'] = pretrained_cfg.pop('labels')
        hf_config['pretrained_cfg'] = pretrained_cfg

    # NOTE currently discarding parent config as only arch name and pretrained_cfg used in timm right now
    pretrained_cfg = hf_config['pretrained_cfg']
    pretrained_cfg['hf_hub_id'] = model_id  # insert hf_hub id for pretrained weight load during model creation
    pretrained_cfg['source'] = 'hf-hub'

    # model should be created with base config num_classes if its exist
    if 'num_classes' in hf_config:
        pretrained_cfg['num_classes'] = hf_config['num_classes']

    # label meta-data in base config overrides saved pretrained_cfg on load
    if 'label_names' in hf_config:
        pretrained_cfg['label_names'] = hf_config.pop('label_names')
    if 'label_descriptions' in hf_config:
        pretrained_cfg['label_descriptions'] = hf_config.pop('label_descriptions')

    model_args = hf_config.get('model_args', {})
    model_name = hf_config['architecture']
    return pretrained_cfg, model_name, model_args

from timm.layers import set_layer_config
from timm.models import is_model, model_entrypoint, load_checkpoint

def split_model_name_tag(model_name: str, no_tag: str = ''):
    model_name, *tag_list = model_name.split('.', 1)
    tag = tag_list[0] if tag_list else no_tag
    return model_name, tag

from urllib.parse import urlsplit

def parse_model_name(model_name: str):
    if model_name.startswith('hf_hub'):
        # NOTE for backwards compat, deprecate hf_hub use
        model_name = model_name.replace('hf_hub', 'hf-hub')
    parsed = urlsplit(model_name)
    assert parsed.scheme in ('', 'timm', 'hf-hub')
    if parsed.scheme == 'hf-hub':
        # FIXME may use fragment as revision, currently `@` in URI path
        return parsed.scheme, parsed.path
    else:
        model_name = os.path.split(parsed.path)[-1]
        return 'timm', model_name


def create_model():
    model_name = 'hf_hub:prov-gigapath/prov-gigapath'
    model_source, model_name = parse_model_name(model_name)
    pretrained_cfg, model_name, model_args = load_model_config_from_hf(model_name)
    kwargs = {}
    if model_args:
        for k, v in model_args.items():
            kwargs.setdefault(k, v)
    create_fn = model_entrypoint(model_name)
    with set_layer_config(scriptable=None, exportable=None, no_jit=None):
        model = create_fn(
            pretrained=False,
            pretrained_cfg=pretrained_cfg,
            pretrained_cfg_overlay=None,
            **kwargs
        )
    load_checkpoint(model, f'{DATA_DIR}/assets/ProvGigaPath/pytorch_model.bin')

    return model

print('before loading backbones ', psutil.virtual_memory().used/1024/1024/1024, "GB")
models_dict = {}
for search_backbone in backbones:
    models_dict[search_backbone] = {}
    if search_backbone == 'HERE_PLIP':
        models_dict[search_backbone]['feature_extractor'] = CLIPModel.from_pretrained(f"{DATA_DIR}/assets/vinid_plip")
        models_dict[search_backbone]['image_processor_or_transform'] = CLIPProcessor.from_pretrained(f"{DATA_DIR}/assets/vinid_plip")
        models_dict[search_backbone]['attention_model'] = AttentionModel()
        models_dict[search_backbone]['state_dict'] = torch.load(f"{DATA_DIR}/assets/snapshot_66_HERE_PLIP.pt", map_location='cpu', weights_only=True)
    elif search_backbone == 'HERE_CONCH':
        from conch.open_clip_custom import create_model_from_pretrained
        models_dict[search_backbone]['feature_extractor'], models_dict[search_backbone]['image_processor_or_transform'] = create_model_from_pretrained('conch_ViT-B-16', checkpoint_path=f'{DATA_DIR}/assets/CONCH_weights_pytorch_model.bin')
        models_dict[search_backbone]['attention_model'] = AttentionModel(backbone='CONCH')
        models_dict[search_backbone]['state_dict'] = torch.load(f"{DATA_DIR}/assets/snapshot_53_HERE_CONCH.pt", map_location='cpu', weights_only=True)
    elif search_backbone == 'HERE_ProvGigaPath':
        models_dict[search_backbone]['feature_extractor'] = create_model()
        models_dict[search_backbone]['image_processor_or_transform'] = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        models_dict[search_backbone]['attention_model'] = AttentionModel(backbone='ProvGigaPath')
        models_dict[search_backbone]['state_dict'] = torch.load(f"{DATA_DIR}/assets/snapshot_39_HERE_ProvGigaPath.pt", map_location='cpu', weights_only=True)
    elif search_backbone == 'HERE_UNI':
        models_dict[search_backbone]['feature_extractor'] = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        models_dict[search_backbone]['feature_extractor'].load_state_dict(torch.load(f"{DATA_DIR}/assets/UNI_pytorch_model.bin", map_location="cpu", weights_only=True), strict=True)
        models_dict[search_backbone]['image_processor_or_transform'] = create_transform(**resolve_data_config(models_dict[search_backbone]['feature_extractor'].pretrained_cfg, model=models_dict[search_backbone]['feature_extractor']))
        models_dict[search_backbone]['attention_model'] = AttentionModel(backbone='UNI')
        models_dict[search_backbone]['state_dict'] = torch.load(f"{DATA_DIR}/assets/snapshot_58_HERE_UNI.pt", map_location='cpu', weights_only=True)
    
    models_dict[search_backbone]['feature_extractor'].eval()
    models_dict[search_backbone]['attention_model'].load_state_dict(models_dict[search_backbone]['state_dict']['MODEL_STATE'], strict=False)
    models_dict[search_backbone]['attention_model'].eval()

print('after loading backbones ', psutil.virtual_memory().used/1024/1024/1024, "GB")

faiss_indexes = {}
faiss_types = ['faiss_IndexHNSWFlat_m32_IVFPQ_nlist128_m8', 'faiss_IndexFlatL2']
for backbone in backbones:
    faiss_indexes[backbone] = {}
    for faiss_type in faiss_types:
        faiss_indexes[backbone][faiss_type] = {}
        for project_name in project_names:
            faiss_bin_filename = f"{DATA_DIR}/assets/faiss_bins/all_data_feat_before_attention_feat_{faiss_type}_{project_name}_{backbone}.bin"
            if os.path.exists(faiss_bin_filename):
                faiss_indexes[backbone][faiss_type][project_name] = faiss.read_index(faiss_bin_filename)

print('after loading faiss indexes ', psutil.virtual_memory().used/1024/1024/1024, "GB")

with open(f'{DATA_DIR}/assets/randomly_1000_data_with_PLIP_ProvGigaPath_CONCH_20241219.pkl', 'rb') as fp: # with HERE_ProvGigaPath, with CONCH, random normal distribution
    randomly_1000_data = pickle.load(fp)

font = ImageFont.truetype("Gidole-Regular.ttf", size=36)
print('before loading lmdb ', psutil.virtual_memory().used/1024/1024/1024, "GB")

# setup LMDB
txns = {}
for project_name in project_names:
    txns[project_name] = []
    dirs = glob.glob(
        f'{DATA_DIR}/lmdb_images_20240622_64/all_images_{project_name}_split*')
    for i in range(len(dirs)):
        lmdb_path = f'{DATA_DIR}/lmdb_images_20240622_64/all_images_{project_name}_split{i}/data'
        env_i = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False,
                          map_async=True)
        txn_i = env_i.begin(write=False)
        txns[project_name].append(txn_i)


def new_web_annotation2(cluster_label, min_dist, x, y, w, h, annoid_str):
    anno = {
        "type": "Annotation",
        "body": [{
            "type": "TextualBody",
            "value": "{}".format(min_dist),
            "purpose": "tagging"
        }],
        "target": {
            "source": "http://localhost:3000/",
            "selector": {
                "type": "FragmentSelector",
                "conformsTo": "http://www.w3.org/TR/media-frags/",
                "value": f"xywh=pixel:{x},{y},{w},{h}"
            }
        },
        "@context": "http://www.w3.org/ns/anno.jsonld",
        "id": annoid_str
    }
    return anno


def knn_search_images_by_faiss(query_embedding, k=10, search_project="ALL", search_method='faiss', search_backbone='HERE_PLIP'):
    if search_project == 'ALL':
        Ds, Is = {}, {}
        for iiiii, project_name in enumerate(project_names):
            Di, Ii = faiss_indexes[search_backbone][search_method][project_name].search(query_embedding, k)
            Di = np.array([dd for dd, ii in zip(Di[0], Ii[0]) if ii>=0])
            beginid = project_start_ids[project_name]
            Ii = [beginid+ii for ii in Ii[0] if ii>=0]
            Ds[project_name] = Di
            Is[project_name] = Ii

        D = np.concatenate(list(Ds.values()))
        I = np.concatenate(list(Is.values()))
        if 'HNSW' in search_method or 'IndexFlatL2' in search_method:
            inds = np.argsort(D)[:k]
        else:  # IP or cosine similarity, the larger, the better
            inds = np.argsort(D)[::-1][:k]
        return D[inds], I[inds]

    else:
        Di, Ii = faiss_indexes[search_backbone][search_method][search_project].search(query_embedding, k)

        Di = np.array([dd for dd, ii in zip(Di[0], Ii[0]) if ii>=0])
        beginid = project_start_ids[search_project]
        Ii = np.array([beginid+ii for ii in Ii[0] if ii>=0])
        return Di, Ii


def compute_mean_std_cosine_similarity_from_random1000_bak(query_embedding, search_project='ALL', search_backbone='HERE_PLIP'):
    distances = 1 - pairwise_distances(query_embedding.reshape(1, -1),
                                       randomly_1000_data[search_backbone][search_project if search_project in randomly_1000_data[search_backbone].keys() else 'ALL'],
                                       metric='cosine')[0]
    return np.mean(distances), np.std(distances), distances
def compute_mean_std_cosine_similarity_from_random1000(query_embedding, search_project='ALL', search_backbone='HERE_PLIP'):
    distances = pairwise_distances(query_embedding.reshape(1, -1), randomly_1000_data[search_backbone][search_project if search_project in randomly_1000_data[search_backbone].keys() else 'ALL'])[0]
    return np.mean(distances), np.std(distances), distances


def get_image_patches(image, sizex=256, sizey=256):
    # w = 2200
    # h = 2380
    # sizex, sizey = 256, 256
    w, h = image.size
    if w < sizex:
        image1 = Image.new(image.mode, (sizey, h), (0, 0, 0))
        image1.paste(image, ((sizex - w) // 2, 0))
        image = image1
    w, h = image.size
    if h < sizey:
        image1 = Image.new(image.mode, (w, sizex), (0, 0, 0))
        image1.paste(image, (0, (sizey - h) // 2))
        image = image1
    w, h = image.size
    # creating new Image object
    image_shown = image.copy()
    img1 = ImageDraw.Draw(image_shown)

    num_x = np.floor(w / sizex)
    num_y = np.floor(h / sizey)
    box_w = int(num_x * sizex)
    box_y = int(num_y * sizey)
    startx = w // 2 - box_w // 2
    starty = h // 2 - box_y // 2
    patches = []
    r = 5
    patch_coords = []
    for x1 in range(startx, w, sizex):
        x2 = x1 + sizex
        if x2 > w:
            continue
        for y1 in range(starty, h, sizey):
            y2 = y1 + sizey
            if y2 > h:
                continue
            img1.line((x1, y1, x1, y2), fill="white", width=1)
            img1.line((x1, y2, x2, y2), fill="white", width=1)
            img1.line((x2, y2, x2, y1), fill="white", width=1)
            img1.line((x2, y1, x1, y1), fill="white", width=1)
            cx, cy = x1 + sizex // 2, y1 + sizey // 2
            patch_coords.append((cx, cy))
            img1.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 0, 0, 0))
            patches.append(image.crop((x1, y1, x2, y2)))
    return patches, patch_coords, image_shown


def get_query_embedding(img_urls, resize=0, search_backbone='HERE_PLIP'):
    image_patches_all = []
    patch_coords_all = []
    image_shown_all = []
    minWorH = 1e8
    sizex, sizey = 256, 256
    # if 'CONCH' in search_backbone:
    #     sizex, sizey = 512, 512
    image_urls_all = {}
    results_dict = {}
    for img_url in img_urls:
        if img_url[:4] == 'http':
            image = Image.open(img_url.replace(
                'https://hidare-dev.ccr.cancer.gov/', '')).convert('RGB')
        elif img_url[:4] == 'data':
            image_data = re.sub('^data:image/.+;base64,', '', img_url)
            image = Image.open(io.BytesIO(
                base64.b64decode(image_data))).convert('RGB')
        else:
            image = Image.open(img_url).convert('RGB')

        W, H = image.size
        minWorH = min(min(W, H), minWorH)
        if W>2048 or H>2048 or W<128 or H<128:
            return None, image_urls_all, results_dict, minWorH

        if 0 < resize:
            resize_scale = 1. / 2**resize
            newW, newH = int(W*resize_scale), int(H*resize_scale)
            minWorH = min(min(newW, newH), minWorH)
            image = image.resize((newW, newH))
        if search_backbone == 'ProvGigaPath':
            image = image.resize((256, 256))
        patches, patch_coords, image_shown = get_image_patches(image, sizex=sizex, sizey=sizey)
        image_patches_all.append(patches)
        patch_coords_all.append(patch_coords)
        image_shown_all.append(image_shown)

    image_patches = [
        patch for patches in image_patches_all for patch in patches]
    with torch.no_grad():

        if search_backbone in ['HERE_PLIP', 'HERE_ProvGigaPath', 'HERE_CONCH', 'HERE_UNI']:
            if search_backbone == 'HERE_PLIP':
                images = models_dict[search_backbone]['image_processor_or_transform'](images=image_patches, return_tensors='pt')['pixel_values']
                feat_after_encoder_feat = models_dict[search_backbone]['feature_extractor'].get_image_features(images).detach()
                # extract feat_before_attention_feat
                embedding = feat_after_encoder_feat @ models_dict[search_backbone]['state_dict']['MODEL_STATE']['attention_net.0.weight'].T + \
                    models_dict[search_backbone]['state_dict']['MODEL_STATE']['attention_net.0.bias']
                # get the attention scores
                results_dict = models_dict[search_backbone]['attention_model'](feat_after_encoder_feat.unsqueeze(0))
            elif search_backbone == 'HERE_ProvGigaPath':
                images = torch.stack([models_dict[search_backbone]['image_processor_or_transform'](example) for example in image_patches])
                feat_after_encoder_feat = models_dict[search_backbone]['feature_extractor'](images).detach()
                # extract feat_before_attention_feat
                embedding = feat_after_encoder_feat @ models_dict[search_backbone]['state_dict']['MODEL_STATE']['attention_net.0.weight'].T + \
                    models_dict[search_backbone]['state_dict']['MODEL_STATE']['attention_net.0.bias']
                # get the attention scores
                results_dict = models_dict[search_backbone]['attention_model'](feat_after_encoder_feat.unsqueeze(0))
            elif search_backbone == 'HERE_CONCH':
                images = torch.stack([models_dict[search_backbone]['image_processor_or_transform'](example) for example in image_patches])
                feat_after_encoder_feat = models_dict[search_backbone]['feature_extractor'].encode_image(images, proj_contrast=False, normalize=False).detach()
                # extract feat_before_attention_feat
                embedding = feat_after_encoder_feat @ models_dict[search_backbone]['state_dict']['MODEL_STATE']['attention_net.0.weight'].T + \
                    models_dict[search_backbone]['state_dict']['MODEL_STATE']['attention_net.0.bias']
                # get the attention scores
                results_dict = models_dict[search_backbone]['attention_model'](feat_after_encoder_feat.unsqueeze(0))
            elif search_backbone == 'HERE_UNI':
                images = torch.stack([models_dict[search_backbone]['image_processor_or_transform'](example) for example in image_patches])
                feat_after_encoder_feat = models_dict[search_backbone]['feature_extractor'](images).detach()
                # extract feat_before_attention_feat
                embedding = feat_after_encoder_feat @ models_dict[search_backbone]['state_dict']['MODEL_STATE']['attention_net.0.weight'].T + \
                    models_dict[search_backbone]['state_dict']['MODEL_STATE']['attention_net.0.bias']
                # get the attention scores
                results_dict = models_dict[search_backbone]['attention_model'](feat_after_encoder_feat.unsqueeze(0))
            # weighted the features using attention scores
            embedding = torch.mm(results_dict['A'], embedding)
            # embedding = results_dict['global_feat'].detach().numpy()
            embedding = embedding.detach().numpy()

            if len(image_patches_all) > 1:
                atten_scores = np.split(results_dict['A'].detach().numpy()[0], np.cumsum(
                    [len(patches) for patches in image_patches_all])[:-1])
            else:
                atten_scores = [results_dict['A'].detach().numpy()[0]]
            for ii, atten_scores_ in enumerate(atten_scores):
                I1 = ImageDraw.Draw(image_shown_all[ii])
                for jj, score in enumerate(atten_scores_):
                    I1.text(patch_coords_all[ii][jj], "{:.4f}".format(
                        score), fill=(0, 255, 255), font=font)

                img_byte_arr = io.BytesIO()
                image_shown_all[ii].save(img_byte_arr, format='JPEG')
                image_urls_all[str(ii)] = "data:image/jpeg;base64, " + \
                    base64.b64encode(img_byte_arr.getvalue()).decode()
        else:
            return None, image_urls_all, results_dict, minWorH

    embedding = embedding.reshape(1, -1)
    embedding /= np.linalg.norm(embedding)
    return embedding, image_urls_all, results_dict, minWorH



def image_search_main(params):
    k = 100
    if 'k' in params:
        k = int(float(params['k']))
    if k <= 0:
        k = 10
    if k >= 500:
        k = 500
    search_project = 'ALL'
    if 'search_project' in params:
        search_project = params['search_project']
    search_method = 'faiss'
    if 'search_method' in params:
        search_method = params['search_method']
    if search_project == 'ST':  
        search_method = 'faiss_IndexFlatL2'
    if search_method == 'faiss_IndexFlatL2':
        search_project = 'ST'

    for key, value in params.items():
        if key != 'img_urls':
            print(key, value)

    start = time.perf_counter()

    query_embedding, images_shown_urls, results_dict, minWorH = \
        get_query_embedding(params['img_urls'], resize=int(float(params['resize'])), search_backbone=params['search_backbone'])  # un-normalized
    
    if query_embedding is None:
        return {'current': 100, 'total': 100, 'status': 'Task completed!',
            'result': {'coxph_html_dict': {}, 'response': {}, 'ranks': {}, 'pred_str': "wrong",
            'images_shown_urls': {}, 'minWorH': minWorH}}

    query_embedding = query_embedding.reshape(1, -1)

    coxph_html_dict = {}

    query_embedding /= np.linalg.norm(query_embedding)  # l2norm normalized

    D, I = knn_search_images_by_faiss(query_embedding,
                                      k=k, search_project=search_project,
                                      search_method=search_method,
                                      search_backbone=params['search_backbone'])
    
    random1000_mean, random1000_std, random1000_dists = compute_mean_std_cosine_similarity_from_random1000(
        query_embedding, search_project=search_project, search_backbone=params['search_backbone'])

    final_response = {}
    final_response1 = {}
    # iinds = np.argsort(I)
    # D = D[iinds].tolist()
    # I = I[iinds].tolist()
    db_conn, db_cursor = get_db()

    sql = f'select a.*, b.scale, b.patch_size_vis_level, b.svs_prefix, b.external_link, b.note from '\
        'faiss_table_20241219 as a, image_table_20241219 as b where '\
            'a.rowid in %s and a.svs_prefix_id = b.svs_prefix_id and '\
                'a.project_id = b.project_id'
    try:
        db_cursor.execute(sql, ([ind + 1 for ind in I],))
    except:
        db_conn, db_cursor = get_db()
        db_cursor.execute(sql, ([ind + 1 for ind in I],))

    res = db_cursor.fetchall()
    db_conn.close()
    infos = {int(item[0]) - 1: item for item in res}
    del res
    gc.collect()

    index_rank_scores = np.arange(1, 1+len(D))[::-1]
    for ii, (score, ind) in enumerate(zip(D, I)):

        if ind not in infos:
            print('not in there')
            continue

        rowid, x, y, svs_prefix_id, proj_id, scale, patch_size_vis_level, slide_name, external_link, note = infos[ind]
        scale = float(scale)
        patch_size_vis_level = int(patch_size_vis_level)
        if len(note) == 0:
            note = 'No clinical information. '

        item = {'_score': score,
                '_zscore': (score - random1000_mean) / random1000_std,
                '_pvalue': len(np.where(random1000_dists <= score)[0]) / len(random1000_dists)}
        project_name = project_names[proj_id]
        x0, y0 = int(x), int(y)
        image_id = '{}_{}_x{:d}_y{:d}'.format(proj_id, svs_prefix_id, x, y)
        image_name = '{}_x{}_y{}'.format(slide_name, x, y)

        if 'ST' in project_name:
            has_gene = '1'
        else:
            has_gene = '0'

        image_id_bytes = image_id.encode('ascii')
        img_bytes = None
        for i in range(len(txns[project_name])):
            if img_bytes is None:
                img_bytes = txns[project_name][i].get(image_id_bytes)

        if img_bytes is None:
            print('no img_bytes')
            continue

        im = Image.open(io.BytesIO(img_bytes))
        buffer = io.BytesIO()
        im.save(buffer, format="jpeg")
        encoded_image = base64.b64encode(buffer.getvalue()).decode()
        img_url = "data:image/jpeg;base64, " + encoded_image

        x = int(float(scale) * float(image_name.split('_')[-2].replace('x', '')))
        y = int(float(scale) * float(image_name.split('_')[-1].replace('y', '')))

        if slide_name in final_response:
            final_response[slide_name]['images'].append(
                {'img_url': img_url, 'x': x, 'y': y, 'x0': x0, 'y0': y0, 'image_name': image_name,
                 'has_gene': has_gene})
            final_response[slide_name]['annotations'].append(
                new_web_annotation2(0, "{:.3f}, z{:.3f}, p{:.3f}".format(item['_score'], item['_zscore'],
                                                                         item['_pvalue']),
                                    x, y, patch_size_vis_level, patch_size_vis_level, ""))
            final_response[slide_name]['scores'].append(float(item['_score']))
            final_response[slide_name]['zscores'].append(
                float(item['_zscore']))
            final_response[slide_name]['note'] = note
            final_response[slide_name]['external_link'] = external_link
            final_response1[slide_name]['index_rank_scores'].append(
                index_rank_scores[ii]
            )
        else:
            final_response[slide_name] = {}
            final_response[slide_name]['project_name'] = project_name if 'KenData' not in project_name else "NCIData"
            final_response[slide_name]['images'] = []
            final_response[slide_name]['images'].append(
                {'img_url': img_url, 'x': x, 'y': y, 'x0': x0, 'y0': y0, 'image_name': image_name,
                 'has_gene': has_gene})
            final_response[slide_name]['annotations'] = []
            final_response[slide_name]['annotations'].append(
                new_web_annotation2(0, "{:.3f}, z{:.3f}, p{:.3f}".format(item['_score'], item['_zscore'],
                                                                         item['_pvalue']),
                                    x, y, patch_size_vis_level, patch_size_vis_level, ""))
            final_response[slide_name]['scores'] = []
            final_response[slide_name]['scores'].append(float(item['_score']))
            final_response[slide_name]['zscores'] = []
            final_response[slide_name]['zscores'].append(
                float(item['_zscore']))
            final_response[slide_name]['note'] = note
            final_response[slide_name]['external_link'] = external_link
            final_response1[slide_name] = {}
            final_response1[slide_name]['index_rank_scores'] = [index_rank_scores[ii]]
    end = time.perf_counter()
    time_elapsed_ms = (end - start) * 1000

    zscore_sum_list = []
    for k in final_response.keys():
        final_response[k]['min_score'] = float(min(final_response[k]['scores']))
        final_response[k]['max_score'] = float(max(final_response[k]['scores']))
        final_response[k]['min_zscore'] = float(min(final_response[k]['zscores']))
        final_response[k]['max_zscore'] = float(max(final_response[k]['zscores']))
        zscore_sum = float(sum(final_response[k]['zscores']))
        # final_response[k]['zscore_sum'] = zscore_sum
        final_response[k]['zscore_sum'] = float(sum(final_response1[k]['index_rank_scores']))
        # zscore_sum_list.append(abs(zscore_sum))
        # zscore_sum_list.append(len(final_response[k]['zscores']))
        zscore_sum_list.append(sum(final_response1[k]['index_rank_scores']))
        final_response[k]['_random1000_mean'] = float(random1000_mean)
        final_response[k]['_random1000_std'] = float(random1000_std)
        final_response[k]['_time_elapsed_ms'] = float(time_elapsed_ms)
    sort_inds = np.argsort(zscore_sum_list)[::-1].tolist()
    allkeys = list(final_response.keys())
    ranks = {rank: allkeys[ind] for rank, ind in enumerate(sort_inds)} # sorted by zscore_sum descend order

    # prediction
    if params['search_backbone'] in ['HERE_PLIP', 'HERE_ProvGigaPath', 'HERE_CONCH', 'HERE_UNI']:
        table_str = [
            '<table border="1"><tr><th>task</th><th>prediction</th></tr>']
        for k,v in models_dict[params['search_backbone']]['attention_model'].classification_dict.items():
            Y_prob_k = F.softmax(results_dict[k + '_logits'], dim=1).detach().numpy()[0]
            table_str.append(
                '<tr><td>{}</td><td>{}: {:.3f}</td></tr>'.format(k.replace('_cls', ''), v[1], Y_prob_k[1]))
        for k in models_dict[params['search_backbone']]['attention_model'].regression_list:
            table_str.append(
                '<tr><td>{}</td><td>{:.3f}</td></tr>'.format(k, results_dict[k + '_logits'].item()))
        table_str.append('</table>')
        pred_str = ''.join(table_str)
    else:
        pred_str = 'None'

    gc.collect()
    return {'current': 100, 'total': 100, 'status': 'Task completed!',
            'result': {'coxph_html_dict': coxph_html_dict, 'response': final_response, 'ranks': ranks, 'pred_str': pred_str,
            'images_shown_urls': images_shown_urls, 'minWorH': minWorH}}

import sys,os,json,glob
import pandas as pd
import numpy as np
import tarfile
import io
from scipy.stats import rankdata
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
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
import timm
import pickle
from sklearn import cluster, neighbors
import time
from PIL import Image, ImageDraw, ImageFont
import openslide
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.middleware.proxy_fix import ProxyFix

from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.use('agg')
import cv2
import pyvips
import multiprocessing
from celery import Celery
import random
import pymysql

# (CHANGE THIS)
DB_USER = "username"
DB_PASSWORD = "password"
DB_HOST = "mysql host ip"
DB_DATABASE = "hidare_app"  

TCGA_COMBINED = '_20240619'
BACKBONE="HiDARE_PLIP"
HIDARE_VERSION = '_20240208'
DATA_DIR = f'data_{BACKBONE}{HIDARE_VERSION}'
with open(f'{DATA_DIR}/assets/args.pkl', 'rb') as fp:
    argsdata = pickle.load(fp)
with open(f'{DATA_DIR}/assets/all_scales{TCGA_COMBINED}.pkl', 'rb') as fp:
    all_scales = pickle.load(fp)
with open(f'{DATA_DIR}/assets/all_notes_dict.pkl', 'rb') as fp:
    all_notes_bak = pickle.load(fp)
all_notes = {}
for k, v in all_notes_bak.items():
    all_notes[k] = v
    if isinstance(k, int):
        all_notes[str(k)] = v
del all_notes_bak

internal_project_names = []  
tcga_names = ['TCGA-COMBINED']
project_names = ['TCGA-COMBINED', 'KenData', 'ST']  # do not change the order   

def get_db():
    db_conn = pymysql.connect(user=DB_USER, password=DB_PASSWORD, host=DB_HOST, database=DB_DATABASE)
    db_conn.autocommit = False
    db_cursor = db_conn.cursor()
    return db_conn, db_cursor

app = Flask(__name__,
            static_url_path='',
            static_folder='',
            template_folder='')
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)


feature_extractor = None
config = None
transform = None
state_dict = None
attention_model = AttentionModel(
    classification_dict=argsdata['classification_dict'],
    regression_list=argsdata['regression_list'],
    args=argsdata['args']
)
if BACKBONE == 'HiDARE_PLIP':
    from transformers import CLIPModel, CLIPProcessor
    feature_extractor = CLIPModel.from_pretrained('vinid/plip')
    image_processor = CLIPProcessor.from_pretrained('vinid/plip')
    feature_extractor = feature_extractor.eval()

    # encoder + attention except final
    state_dict = torch.load(f"{DATA_DIR}/assets/snapshot_95.pt", map_location='cpu')
    attention_model.load_state_dict(state_dict['MODEL_STATE'], strict=False)
    attention_model = attention_model.eval()

    ST_ROOT = f'{DATA_DIR}/20240202v4_ST/PanCancer2GPUsFP/shared_attention_imagenetPLIP/split1_e95_h224_density_vis/feat_before_attention_feat/test/analysis/one_patient_top_128'

with open(f'{DATA_DIR}/assets/faiss_bins/faiss_infos{TCGA_COMBINED}.pkl', 'rb') as fp:
    tmpdata = pickle.load(fp)
    all_project_ids = tmpdata['all_project_ids']
    project_names_dict = tmpdata['project_names_dict']
    svs_prefixes_dict = tmpdata['svs_prefixes_dict']

faiss_Ms = [32]
faiss_nlists = [128]
faiss_indexes = {'faiss_IndexFlatIP': {}}
for project_name in ['ST']:  # project_names:  # only ST support IndexFlatIP search
    faiss_indexes['faiss_IndexFlatIP'][project_name] = \
        faiss.read_index(
            f"{DATA_DIR}/assets/faiss_bins/all_data_feat_before_attention_feat_faiss_IndexFlatIP_{project_name}.bin")
for m in faiss_Ms:
    for nlist in faiss_nlists:
        faiss_indexes[f'faiss_IndexHNSWFlat_m{m}_IVFPQ_nlist{nlist}_m8'] = {}
        for project_name in project_names:
            faiss_indexes[f'faiss_IndexHNSWFlat_m{m}_IVFPQ_nlist{nlist}_m8'][project_name] = \
                faiss.read_index(
                    f"{DATA_DIR}/assets/faiss_bins/all_data_feat_before_attention_feat_faiss_IndexHNSWFlat_m{m}_IVFPQ_nlist{nlist}_m8_{project_name}.bin")


with open(f'{DATA_DIR}/assets/randomly_background_samples_for_train.pkl', 'rb') as fp:
    randomly_1000_data = pickle.load(fp)

with open('gene_names.csv', 'r') as fp:
    valid_gene_names = set([line.strip() for line in fp.readlines()])

if 'ALL' not in randomly_1000_data:
    tmp_keys = []
    tmp_embeddings = []
    tmp_embeddings_tcga = []
    for project_name in randomly_1000_data.keys():  # project_names:
        tmp_embeddings.append(randomly_1000_data[project_name]['embeddings'])
        if 'TCGA' in project_name:
            tmp_embeddings_tcga.append(randomly_1000_data[project_name]['embeddings'])
    randomly_1000_data['ALL'] = {
        'keys': tmp_keys, 'embeddings': np.concatenate(tmp_embeddings, axis=0)}
    randomly_1000_data['TCGA-ALL'] = {
        'keys': tmp_keys, 'embeddings': np.concatenate(tmp_embeddings_tcga, axis=0)}
    del tmp_embeddings, tmp_embeddings_tcga

font = ImageFont.truetype("Gidole-Regular.ttf", size=36)


lmdb_path = f'{DATA_DIR}/assets/lmdb_ST_gene_exps/data'
if os.path.exists(lmdb_path):
    env_gene = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False,
                         map_async=True)
    txn_gene = env_gene.begin(write=False)
else:
    txn_gene = None

# get the filelist of gene_counts
valid_ST_prefixes_filename = f'{DATA_DIR}/assets/lmdb_ST_gene_exps/valid_ST_prefixes.txt'
if os.path.exists(valid_ST_prefixes_filename):
    with open(valid_ST_prefixes_filename, 'r') as fp:
        gene_counts_prefixes = set([line.strip() for line in fp.readlines()])
else:
    gene_counts_prefixes = {}

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

# from https://github.com/mahmoodlab/CLAM
def _assertLevelDownsamples(slide):
    level_downsamples = []
    dim_0 = slide.level_dimensions[0]

    for downsample, dim in zip(slide.level_downsamples, slide.level_dimensions):
        estimated_downsample = (dim_0[0] / float(dim[0]), dim_0[1] / float(dim[1]))
        level_downsamples.append(estimated_downsample) if estimated_downsample != (
            downsample, downsample) else level_downsamples.append((downsample, downsample))

    return level_downsamples


def _assertLevelDownsamplesV2(level_dimensions, level_downsamples):
    level_downsamples_new = []
    dim_0 = level_dimensions[0]

    for downsample, dim in zip(level_downsamples, level_dimensions):
        estimated_downsample = (
            dim_0[0] / float(dim[0]), dim_0[1] / float(dim[1]))
        level_downsamples_new.append(estimated_downsample) if estimated_downsample != (
            downsample, downsample) else level_downsamples_new.append((downsample, downsample))

    return level_downsamples_new


def to_percentiles(scores):
    scores = rankdata(scores, 'average') / len(scores) * 100
    return scores


def block_blending(slide, img, vis_level, top_left, bot_right, alpha=0.5, blank_canvas=False, block_size=1024):
    level_downsamples = _assertLevelDownsamples(slide)
    downsample = level_downsamples[vis_level]
    w = img.shape[1]
    h = img.shape[0]
    block_size_x = min(block_size, w)
    block_size_y = min(block_size, h)

    shift = top_left  # amount shifted w.r.t. (0,0)
    for x_start in range(top_left[0], bot_right[0], block_size_x * int(downsample[0])):
        for y_start in range(top_left[1], bot_right[1], block_size_y * int(downsample[1])):

            # 1. convert wsi coordinates to image coordinates via shift and scale
            x_start_img = int((x_start - shift[0]) / int(downsample[0]))
            y_start_img = int((y_start - shift[1]) / int(downsample[1]))

            # 2. compute end points of blend tile, careful not to go over the edge of the image
            y_end_img = min(h, y_start_img + block_size_y)
            x_end_img = min(w, x_start_img + block_size_x)

            if y_end_img == y_start_img or x_end_img == x_start_img:
                continue

            # 3. fetch blend block and size
            blend_block = img[y_start_img:y_end_img, x_start_img:x_end_img]
            blend_block_size = (x_end_img - x_start_img, y_end_img - y_start_img)

            if not blank_canvas:
                # 4. read actual wsi block as canvas block
                pt = (x_start, y_start)
                canvas = np.array(slide.read_region(pt, vis_level, blend_block_size).convert("RGB"))
            else:
                # 4. OR create blank canvas block
                canvas = np.array(Image.new(size=blend_block_size, mode="RGB", color=(255, 255, 255)))

            # 5. blend color block and canvas block
            img[y_start_img:y_end_img, x_start_img:x_end_img] = cv2.addWeighted(blend_block, alpha, canvas,
                                                                                1 - alpha, 0, canvas)
    return img


# modified from https://github.com/mahmoodlab/CLAM
def visHeatmap(wsi, scores, coords, vis_level=1,
               top_left=None, bot_right=None,
               patch_size=(256, 256),
               blank_canvas=False, canvas_color=(220, 20, 50), alpha=0.4,
               blur=False, overlap=0.0,
               segment=True, use_holes=True,
               convert_to_percentiles=False,
               binarize=False, thresh=0.5,
               max_size=None,
               custom_downsample=1,
               cmap='jet',
               self=None,
               startstep=-1,
               endstep=-1,
               spot_size=-1, stX=None, stY=None,st_patch_size=-1):
    """
    Args:
        scores (numpy array of float): Attention scores
        coords (numpy array of int, n_patches x 2): Corresponding coordinates (relative to lvl 0)
        vis_level (int): WSI pyramid level to visualize
        patch_size (tuple of int): Patch dimensions (relative to lvl 0)
        blank_canvas (bool): Whether to use a blank canvas to draw the heatmap (vs. using the original slide)
        canvas_color (tuple of uint8): Canvas color
        alpha (float [0, 1]): blending coefficient for overlaying heatmap onto original slide
        blur (bool): apply gaussian blurring
        overlap (float [0 1]): percentage of overlap between neighboring patches (only affect radius of blurring)
        segment (bool): whether to use tissue segmentation contour (must have already called self.segmentTissue such that
                        self.contours_tissue and self.holes_tissue are not None
        use_holes (bool): whether to also clip out detected tissue cavities (only in effect when segment == True)
        convert_to_percentiles (bool): whether to convert attention scores to percentiles
        binarize (bool): only display patches > threshold
        threshold (float): binarization threshold
        max_size (int): Maximum canvas size (clip if goes over)
        custom_downsample (int): additionally downscale the heatmap by specified factor
        cmap (str): name of matplotlib colormap to use
    """
    starttime = time.time()

    if vis_level < 0:
        vis_level = wsi.get_best_level_for_downsample(32)

    # downsample = self.level_downsamples[vis_level]
    level_downsamples = _assertLevelDownsamples(wsi)
    downsample = level_downsamples[vis_level]
    scale = [1 / downsample[0], 1 / downsample[1]]  # Scaling from 0 to desired level

    if len(scores.shape) == 2:
        scores = scores.flatten()

    if binarize:
        if thresh < 0:
            threshold = 1.0 / len(scores)

        else:
            threshold = thresh

    else:
        threshold = 0.0

    ##### calculate size of heatmap and filter coordinates/scores outside specified bbox region #####
    if top_left is not None and bot_right is not None:
        scores, coords = screen_coords(scores, coords, top_left, bot_right)
        coords = coords - top_left
        top_left = tuple(top_left)
        bot_right = tuple(bot_right)
        w, h = tuple((np.array(bot_right) * scale).astype(int) - (np.array(top_left) * scale).astype(int))
        region_size = (w, h)

    else:
        # region_size = self.level_dim[vis_level]
        region_size = wsi.level_dimensions[vis_level]
        top_left = (0, 0)
        # bot_right = self.level_dim[0]
        bot_right = wsi.level_dimensions[0]
        w, h = region_size

    patch_size = np.ceil(np.array(patch_size) * np.array(scale)).astype(int)
    coords = np.ceil(coords * np.array(scale)).astype(int)

    ###### normalize filtered scores ######
    if convert_to_percentiles:
        scores = to_percentiles(scores)

    scores /= 100

    ######## calculate the heatmap of raw attention scores (before colormap)
    # by accumulating scores over overlapped regions ######
    if self is not None:
        self.update_state(state='PROGRESS', meta={'current': startstep, 'total': 100, 'status': ''})

    # heatmap overlay: tracks attention score over each pixel of heatmap
    # overlay counter: tracks how many times attention score is accumulated over each pixel of heatmap
    overlay = np.full(np.flip(region_size), 0).astype(float)
    counter = np.full(np.flip(region_size), 0).astype(np.uint16)
    count = 0
    progressstep = 0.25 * (endstep - startstep) / len(coords)
    for idx in range(len(coords)):
        score = scores[idx]
        coord = coords[idx]
        if score >= threshold:
            if binarize:
                score = 1.0
                count += 1
        else:
            score = 0.0
        # accumulate attention
        overlay[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]] += score
        # accumulate counter
        counter[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]] += 1

        if self is not None:
            self.update_state(state='PROGRESS', meta={'current': startstep + idx * progressstep, 'total': 100, 'status': ''})

    if self is not None:
        self.update_state(state='PROGRESS', meta={'current': startstep + 0.25 * (endstep - startstep), 'total': 100, 'status': ''})

    # fetch attended region and average accumulated attention
    zero_mask = counter == 0

    if binarize:
        overlay[~zero_mask] = np.around(overlay[~zero_mask] / counter[~zero_mask])
    else:
        overlay[~zero_mask] = overlay[~zero_mask] / counter[~zero_mask]
    del counter
    if blur:
        overlay = cv2.GaussianBlur(overlay, tuple((patch_size * (1 - overlap)).astype(int) * 2 + 1), 0)

    if not blank_canvas:
        # downsample original image and use as canvas
        img = np.array(wsi.read_region(top_left, vis_level, region_size).convert("RGB"))
    else:
        # use blank canvas
        img = np.array(Image.new(size=region_size, mode="RGB", color=(255, 255, 255)))

        # return Image.fromarray(img) #raw image

    twenty_percent_chunk = max(1, int(len(coords) * 0.2))

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    if self is not None:
        self.update_state(state='PROGRESS', meta={'current': startstep + 0.5 * (endstep - startstep), 'total': 100, 'status': ''})

    progressstep = 0.25 * (endstep - startstep) / len(coords)
    for idx in range(len(coords)):
        score = scores[idx]
        coord = coords[idx]
        if score >= threshold:
            # attention block
            raw_block = overlay[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]]

            img_block = (cmap(raw_block) * 255)[:, :, :3].astype(np.uint8)

            # rewrite image block
            img[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]] = img_block.copy()
        if self is not None:
            self.update_state(state='PROGRESS', meta={'current': startstep + 0.5 * (endstep - startstep) + idx * progressstep, 'total': 100, 'status': ''})

    del overlay

    if self is not None:
        self.update_state(state='PROGRESS', meta={'current': startstep + 0.75 * (endstep - startstep), 'total': 100, 'status': ''})

    if blur:
        img = cv2.GaussianBlur(img, tuple((patch_size * (1 - overlap)).astype(int) * 2 + 1), 0)

    if alpha < 1.0:
        img = block_blending(wsi, img, vis_level, top_left, bot_right, alpha=alpha, blank_canvas=blank_canvas,
                             block_size=1024)

    img = Image.fromarray(img)
    w, h = img.size

    if self is not None:
        self.update_state(state='PROGRESS', meta={'current': startstep + 0.85 * (endstep - startstep), 'total': 100, 'status': ''})
    if spot_size>0 and stX and stY and st_patch_size>0:
        scale = scale[0]
        draw = ImageDraw.Draw(img)
        circle_radius = int(spot_size * scale * 0.5)
        cmap = plt.get_cmap("tab10")
        colors = (np.array(cmap.colors)*255).astype(np.uint8)
        st_patch_size = int(st_patch_size * scale)
        for ind, (x,y) in enumerate(zip(stX, stY)):
            x, y = int(x * scale), int(y * scale)
            xy = [x-circle_radius, y-circle_radius, x+circle_radius, y+circle_radius]
            draw.ellipse(xy, outline=(255, 255, 255), width=3)

    if custom_downsample > 1:
        img = img.resize((int(w / custom_downsample), int(h / custom_downsample)))

    if max_size is not None and (w > max_size or h > max_size):
        resizeFactor = max_size / w if w > h else max_size / h
        img = img.resize((int(w * resizeFactor), int(h * resizeFactor)))

    if self is not None:
        self.update_state(state='PROGRESS', meta={'current': startstep + 0.95 * (endstep - startstep), 'total': 100, 'status': ''})

    return img


def new_web_annotation(cluster_label, min_dist, x, y, w, h, annoid_str):
    anno = {
        "type": "Annotation",
        "body": [{
            "type": "TextualBody",
            "value": "{:d},({:.3f})".format(cluster_label, min_dist),
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


def map_to_global_index(project_name, inds):
    proj_idx = project_names_dict[project_name]
    global_inds = np.where(all_project_ids == proj_idx)[0]
    return global_inds[inds]


def knn_search_images_by_faiss(query_embedding, k=10, search_project="ALL", search_method='faiss', self=None):
    if search_project == 'ALL' or search_project == 'TCGA-ALL':
        project_names_ = project_names if search_project == 'ALL' else tcga_names
        Ds, Is = {}, {}
        
        progressstep = 60 / len(project_names_)

        for iiiii, project_name in enumerate(project_names_):
            if project_name in internal_project_names:
                continue
            if 'Binary' in search_method:
                Di, Ii = faiss_indexes[search_method][project_name]['index'].search(
                    query_embedding, k)
            elif search_method == 'faiss_IndexFlatIP':
                Di, Ii = faiss_indexes['faiss_IndexFlatIP'][project_name].search(
                    query_embedding, k)
            else: # 'HNSW' in search_method:
                Di, Ii = faiss_indexes[search_method][project_name].search(
                    query_embedding, k)
            Ii_filtered = [ii for ii in Ii[0] if ii>=0]
            Di = np.array([dd for dd, ii in zip(Di[0], Ii[0]) if ii>=0])
            Ii = map_to_global_index(project_name, Ii_filtered)
            Ds[project_name] = Di
            Is[project_name] = Ii

            if self is not None:
                self.update_state(state='PROGRESS', meta={'current': 10 + iiiii*progressstep, 'total': 100, 'status': ''})

        D = np.concatenate(list(Ds.values()))
        I = np.concatenate(list(Is.values()))
        if 'Binary' in search_method or 'L2' in search_method or 'HNSW' in search_method:
            inds = np.argsort(D)[:k]
        else:  # IP or cosine similarity, the larger, the better
            inds = np.argsort(D)[::-1][:k]
        return D[inds], I[inds]

    else:
        if 'Binary' in search_method:
            Di, Ii = faiss_indexes[search_method][search_project]['index'].search(
                query_embedding, k)
        elif search_method == 'faiss_IndexFlatIP':
            Di, Ii = faiss_indexes['faiss_IndexFlatIP'][search_project].search(
                query_embedding, k)
        else: # using 'HNSW':
            Di, Ii = faiss_indexes[search_method][search_project].search(
                query_embedding, k)

        if self is not None:
            self.update_state(state='PROGRESS', meta={'current': 10 + 40, 'total': 100, 'status': ''})

        Ii_filtered = [ii for ii in Ii[0] if ii>=0]
        Di = np.array([dd for dd, ii in zip(Di[0], Ii[0]) if ii>=0])
        Ii = map_to_global_index(search_project, Ii_filtered)
        return Di, Ii


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.route('/scale', methods=['POST', 'GET'])
def get_scale():
    params = request.get_json()
    k, v = None, None
    project_name = params['project_name']
    slide_name = params['slide_name']
    key = '{}_{}'.format(project_name, slide_name)
    if key in all_scales:
        scale = all_scales[key]['scale']
        patch_size_vis_level = all_scales[key]['patch_size_vis_level']
    else:
        scale = [1.0, 1.0]
        patch_size_vis_level = 256
    return {'scale': scale, 'patch_size_vis_level': int(patch_size_vis_level)}


@app.route('/')
def index():
    return render_template('main.html')

@app.route('/image_retrieval')
def HE_image_retrieval():
    return render_template('image2image.html')


@app.route('/gene_search_ST')
def gene_search_ST():
    return render_template('gene2image.html')

@app.route('/help')
def hidare_help():
    return render_template('help.html')

@app.route('/contact')
def hidare_contact():
    return render_template('contact.html')

@app.route('/download_gene', methods=['POST', 'GET'])
def download_gene():
    items = request.get_json()
    df = None
    for item in items:
        project_name = item['project_name']
        slide_name = item['slide_name']

        gene_name_id = '{}_{}_gene_names'.format(project_name, slide_name)
        gene_names_bytes = txn_gene.get(gene_name_id.encode('ascii'))
        gene_names = json.loads(gene_names_bytes.decode('ascii'))['gene_names']
        coords = item['coords'].split(',')
        csv_content = {'gene_name': gene_names}
        for j in range(0, len(coords) - 1, 2):
            x = int(float(coords[j]))
            y = int(float(coords[j + 1]))

            image_id = '{}_{}_x{}_y{}'.format(project_name, slide_name, x, y)

            gene_bytes = txn_gene.get(image_id.encode('ascii'))

            if gene_bytes:
                gene = np.frombuffer(
                    gene_bytes, dtype=np.float32).reshape(2, -1)
                gene_vec = gene[0, :]
                csv_content[image_id] = ['{:.6f}'.format(v) for v in gene_vec]
        df1 = pd.DataFrame(csv_content)
        df1 = df1.set_index('gene_name')
        if df is None:
            df = df1
        else:
            df = df.join(df1, how='outer')

    if df is not None:
        return {'gene_csv': df.to_csv()}
    else:
        return {'gene_csv': 'no gene data found\n'}


@app.route('/download_patches2', methods=['POST', 'GET'])
def download_patches2():

    items = request.get_json()
    json_filename = items['images_dir']
    cname = items['cname']
    slide_name = os.path.basename(json_filename).replace('_cs.json', '')

    fh = io.BytesIO()
    tar_fp = tarfile.open(fileobj=fh, mode='w:gz')

    with open(json_filename, 'r') as fp:
        dd = json.load(fp)

    for line in dd[cname].strip().split('\n'):
        splits = line.split(',')
        patch = Image.open(os.path.join(os.path.dirname(json_filename), '..', '..', '..', '..',
                           'patch_images', slide_name, 'x{}_y{}.JPEG'.format(splits[-2], splits[-1]))).convert('RGB')
        im_buffer = io.BytesIO()
        patch.save(im_buffer, format='JPEG')
        info = tarfile.TarInfo(
            name="{}/{}/x{}_y{}.JPEG".format(slide_name, cname, splits[-2], splits[-1]))
        info.size = im_buffer.getbuffer().nbytes
        info.mtime = time.time()
        im_buffer.seek(0)
        tar_fp.addfile(info, im_buffer)

    tar_fp.close()
    savedir = '{}/temp_images_dir/{}/'.format(DATA_DIR, time.time())
    os.makedirs('{}'.format(savedir), exist_ok=True)
    filepath = '{}/{}_{}.tar.gz'.format(savedir, slide_name, cname)
    with open(filepath, 'wb') as fp:
        fp.write(fh.getvalue())
    return {'filepath': filepath}


@app.route('/gene_search_v2_celery', methods=['POST', 'GET'])
def gene_search_v2_celery():

    params = request.get_json()
    task = gene_search_v2_celery_main.apply_async(args=(params,))
    return jsonify({}), 202, {'Location': url_for('gene_search_v2_celery_taskstatus', task_id=task.id)}


@app.route('/status/<task_id>')
def gene_search_v2_celery_taskstatus(task_id):
    task = gene_search_v2_celery_main.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', '')
        }
        if 'result' in task.info:
            response['result'] = task.info['result']
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),  # this is the exception raised
        }
    return jsonify(response)


@celery.task(bind=True)
def gene_search_v2_celery_main(self, params):

    start = time.perf_counter()

    gene_names = []
    cohensd_thres = None
    try:
        gene_names = [vv for vv in params['gene_names'].split(',') if len(vv) > 0 and vv in valid_gene_names]
        cohensd_thres = float(params['cohensd_thres'])
    except:
        pass

    status = ''
    if len(gene_names) == 0: 
        status = 'Invalid input gene names. \n'
    if cohensd_thres == None:
        cohensd_thres = 1.0
        status += 'Cohensd values should be float. Set it to 1.0. \n'

    status += 'Begin query ... wait...'
    self.update_state(state='PROGRESS', meta={'current': 10, 'total': 100, 'status': status})

    db_conn, db_cursor = get_db()
    hidare_table_str = f'cluster_result_table{HIDARE_VERSION} as a, gene_table{HIDARE_VERSION} as b, cluster_table{HIDARE_VERSION} as c, cluster_setting_table{HIDARE_VERSION} as d, st_table{HIDARE_VERSION} as e'

    if len(gene_names) == 0:
        if cohensd_thres is None:
            sql = f'select a.*, b.symbol, d.cluster_setting, e.prefix, c.cluster_label, c.cluster_info from {hidare_table_str} '\
                'where a.gene_id = b.id and c.id = a.c_id and c.cs_id = d.id and c.st_id = e.id order by a.cohensd desc limit 100;'
            print('sql1', sql, gene_names, cohensd_thres)
            try:
                db_cursor.execute(sql)
            except:
                db_conn, db_cursor = get_db()
                db_cursor.execute(sql)

        else:
            sql = f'select a.*, b.symbol, d.cluster_setting, e.prefix, c.cluster_label, c.cluster_info from {hidare_table_str} '\
                'where a.cohensd > %s and a.gene_id = b.id and c.id = a.c_id and c.cs_id = d.id and c.st_id = e.id order by a.cohensd desc limit 100;'
            print('sql2', sql, gene_names, cohensd_thres)
            try:
                db_cursor.execute(sql, (cohensd_thres,))
            except:
                db_conn, db_cursor = get_db()
                db_cursor.execute(sql, (cohensd_thres,))

    else:
        if cohensd_thres is None:
            sql = f'select a.*, b.symbol, d.cluster_setting, e.prefix, c.cluster_label, c.cluster_info from {hidare_table_str} '\
                'where b.symbol in %s and a.gene_id = b.id and c.id = a.c_id and c.cs_id = d.id and c.st_id = e.id order by a.cohensd desc limit 100;'
            print('sql3', sql, gene_names, cohensd_thres)
            try:
                db_cursor.execute(sql, (gene_names,))
            except:
                db_conn, db_cursor = get_db()
                db_cursor.execute(sql, (gene_names,))
        else:
            sql = f'select a.*, b.symbol, d.cluster_setting, e.prefix, c.cluster_label, c.cluster_info from {hidare_table_str} '\
                'where b.symbol in %s and a.cohensd > %s and a.gene_id = b.id and c.id = a.c_id and c.cs_id = d.id and c.st_id = e.id order by a.cohensd desc limit 100;'
            print('sql4', sql, gene_names, cohensd_thres)
            try:
                db_cursor.execute(sql, (gene_names, cohensd_thres))
            except:
                db_conn, db_cursor = get_db()
                db_cursor.execute(sql, (gene_names, cohensd_thres))

    self.update_state(state='PROGRESS', meta={'current': 50, 'total': 100, 'status': ''})

    result = db_cursor.fetchall()
    db_conn.close()
    if result is None or len(result) == 0:
        return {'current': 100, 'total': 100, 'status': 'Task completed!',
            'result': {'coxph_html_dict': '', 'response': '', 'images_shown_urls': ''}}

    result_df = pd.DataFrame(result, columns=['id', 'c_id', 'gene_id', 'cohensd', 'pvalue', 'pvalue_corrected',
                             'zscore', 'gene_symbol', 'cluster_setting', 'ST_prefix', 'cluster_label', 'cluster_info'])
    removed_columns = ['id', 'c_id', 'gene_id']
    result_df = result_df.drop(columns=removed_columns)

    self.update_state(state='PROGRESS', meta={'current': 70, 'total': 100, 'status': ''})

    final_response = {}
    for slide_name, df_ST in result_df.groupby('ST_prefix'):
        if slide_name not in final_response:
            final_response[slide_name] = {}

            if slide_name in all_notes:
                note = all_notes[slide_name]
            else:
                note = 'No clinical information.'

            final_response[slide_name]['comment'] = note
            final_response[slide_name]['comment_generated'] = ''

    root = ST_ROOT
    notes = []
    notes_geneated = []
    patch_annotations = []
    for rowid, row in result_df.iterrows():
        prefix = row['ST_prefix']
        cluster_setting = row['cluster_setting']
        cluster_label = int(float(row['cluster_label']))
        if True: #try:
            with open(os.path.join(root, cluster_setting, prefix, prefix+'_cluster_data.pkl'), 'rb') as fp:
                cluster_labels = pickle.load(fp)['cluster_labels']
            with open(os.path.join(root, cluster_setting, prefix, prefix+'_patches_annotations.json'), 'r') as fp:
                all_patch_annotations = json.load(fp)
            inds = np.where(cluster_labels == cluster_label)[0]
            patch_annotations.append([all_patch_annotations[indd] for indd in inds])
        # except:
        #     patch_annotations.append([])
        if prefix in final_response:
            notes.append(final_response[prefix]['comment'] )
            notes_geneated.append(final_response[prefix]['comment_generated'] )
        else:
            notes.append('No this information')
            notes_geneated.append('No this information')
    result_df['note'] = notes;
    result_df['note_generated'] = notes_geneated;
    result_df['annotations'] = patch_annotations

    gc.collect()
    self.update_state(state='PROGRESS', meta={'current': 95, 'total': 100, 'status': ''})

    return {'current': 100, 'total': 100, 'status': 'Task completed!',
            'result': {'coxph_html_dict': '', 'response': json.dumps(result_df.to_dict()), 'images_shown_urls': ''}}


def compute_gene_map_dzi(results_path, slide_name, gene_names, gene_map_dir, self=None):

    if len(gene_names) == 0:
        return

    if self is not None:
        self.update_state(state='PROGRESS', meta={'current': 10, 'total': 100, 'status': ''})

    gene_data_filename = f'{results_path}/../../../../gene_data/{slide_name}_gene_data.pkl'

    with open(gene_data_filename, 'rb') as fp:
        gene_data_dict = pickle.load(fp)

    vst_filename = f'{results_path}/../../../../vst_dir/{slide_name}.tsv'
    coord_df = gene_data_dict['coord_df']
    counts_df = gene_data_dict['counts_df']
    barcode_col_name = gene_data_dict['barcode_col_name']
    Y_col_name = gene_data_dict['Y_col_name']
    X_col_name = gene_data_dict['X_col_name']
    mpp = gene_data_dict['mpp']

    vst = pd.read_csv(vst_filename, sep='\t', index_col=0)
    vst = vst.subtract(vst.mean(axis=1), axis=0)
    vst = vst.T
    vst.index.name = 'barcode'

    coord_df1 = coord_df.rename(columns={barcode_col_name: 'barcode', X_col_name: 'X', Y_col_name: 'Y'}).set_index('barcode')
    coord_df1 = coord_df1.loc[vst.index.values]

    stX = coord_df1['X'].values.tolist()
    stY = coord_df1['Y'].values.tolist()
    del coord_df1

    st_patch_size = int(pow(2, np.ceil(np.log(64 / mpp) / np.log(2))))
    st_all_coords = np.array([stX, stY]).T
    st_all_coords -= st_patch_size // 2
    st_all_coords = st_all_coords.astype(np.int32)

    svs_filename = f'{results_path}/../../../../svs/{slide_name}.tif'
    if not os.path.exists(svs_filename):
        prefix1 = slide_name.replace('10x_', '')
        svs_filename = f'{results_path}/../../../../svs/{prefix1}.tif'

    if '10x_' in slide_name:
        with open('{}/../../../../spatial/10x/{}/spatial/scalefactors_json.json'.format(results_path, slide_name.replace('10x_', '')), 'r') as fp:
            spot_size = float(json.load(fp)['spot_diameter_fullres'])
    elif 'TNBC_' in slide_name:
        with open('{}/../../../../spatial/TNBC/{}/spatial/scalefactors_json.json'.format(results_path, slide_name), 'r') as fp:
            spot_size = float(json.load(fp)['spot_diameter_fullres'])
    else:
        spot_size = st_patch_size

    if self is not None:
        self.update_state(state='PROGRESS', meta={'current': 30, 'total': 100, 'status': ''})

    slide = openslide.open_slide(svs_filename)
    dimension = slide.level_dimensions[1] if len(slide.level_dimensions) > 1 else slide.level_dimensions[0]
    if dimension[0] > 100000 or dimension[1] > 100000:
        vis_level = 2
    else:
        vis_level = 1
    if len(slide.level_dimensions) == 1:
        vis_level = 0
    vst.columns = [n.upper() for n in vst.columns]
    progressstep = 60 / len(gene_names)
    for iiii, gene_name in enumerate(gene_names):
        if gene_name not in vst.columns:
            os.system('rm -rf "{}"'.format(os.path.join(gene_map_dir, gene_name+'.doing')))
            with open(os.path.join(gene_map_dir, gene_name+'.error'), 'w') as fp:
                pass
            continue
        data1 = vst[gene_name].values
        lo, hi = np.percentile(data1, (1, 99))
        data1 = 100 * (data1 - lo) / (hi - lo)
        save_filename = os.path.join(gene_map_dir, gene_name)
        img = visHeatmap(slide, scores=data1, coords=st_all_coords,
                            vis_level=vis_level, patch_size=(st_patch_size, st_patch_size),
                            convert_to_percentiles=False, alpha=0.5,
                            self=self,startstep=30+iiii*progressstep, endstep=30+(iiii+1)*progressstep*0.8,
                            spot_size=spot_size, stX=stX, stY=stY,st_patch_size=st_patch_size)
        img_vips = pyvips.Image.new_from_array(img)
        img_vips.dzsave(save_filename, tile_size=1024)
        time.sleep(1)
        del img, img_vips
        os.system('rm -rf "{}"'.format(os.path.join(gene_map_dir, gene_name+'.doing')))
        if self is not None:
            self.update_state(state='PROGRESS', meta={'current': 30+(iiii+1)*progressstep, 'total': 100, 'status': ''})
    
    if self is not None:
        self.update_state(state='PROGRESS', meta={'current': 90, 'total': 100, 'status': ''})





@app.route('/get_gene_map_zip_celery', methods=['POST', 'GET'])
def get_gene_map_zip_celery():

    params = request.get_json()
    task = get_gene_map_zip_celery_main.apply_async(args=(params,))
    return jsonify({}), 202, {'Location': url_for('get_gene_map_zip_celery_taskstatus', task_id=task.id)}


@app.route('/status/<task_id>')
def get_gene_map_zip_celery_taskstatus(task_id):
    task = get_gene_map_zip_celery_main.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', '')
        }
        if 'result' in task.info:
            response['result'] = task.info['result']
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),  # this is the exception raised
        }
    return jsonify(response)


@celery.task(bind=True)
def get_gene_map_zip_celery_main(self, items):
    gene_names = [n.strip().upper() for n in items['gene_names'].split(',')]
    slide_name = items['slide_name']
    results_path = items['results_path']

    gene_map_dir = f'{results_path}/../../../../gene_map_dzi/{slide_name}/'
    if not os.path.exists(gene_map_dir):
        os.makedirs(gene_map_dir, exist_ok=True)
    
    gene_names_to_be_computed = [gene_name for gene_name in gene_names 
        if not os.path.exists(os.path.join(gene_map_dir, gene_name+'.dzi')) 
        and not os.path.exists(os.path.join(gene_map_dir, gene_name+'.doing'))
        and not os.path.exists(os.path.join(gene_map_dir, gene_name+'.error'))
        ]
    
    self.update_state(state='PROGRESS', meta={'current': 5, 'total': 100, 'status': ''})

    if len(gene_names_to_be_computed) > 0:
        for gene_name in gene_names:
            with open(os.path.join(gene_map_dir, gene_name+'.doing'), 'w') as fp:
                pass
        compute_gene_map_dzi(results_path, slide_name, gene_names_to_be_computed, gene_map_dir, self=self)

    self.update_state(state='PROGRESS', meta={'current': 95, 'total': 100, 'status': ''})

    response_dict = {'status': 'doing', 'results': {}}
    alldone = [False for _ in range(len(gene_names))]
    for ii, gene_name in enumerate(gene_names):
        gene_map_name = os.path.join(gene_map_dir, gene_name+'.dzi')
        if os.path.exists(gene_map_name):
            response_dict['results'][gene_name] = gene_map_name
            alldone[ii] = True
        elif os.path.exists(os.path.join(gene_map_dir, gene_name+'.error')):
            alldone[ii] = True
    if np.all(alldone):
        response_dict['status'] = 'alldone'

    # return response_dict
    return {'current': 100, 'total': 100, 'status': 'Task completed!',
            'result': response_dict}


@app.route('/gene', methods=['POST', 'GET'])
def get_gene():
    params = request.get_json()
    project_name = params['project_name']
    slide_name = params['slide_name']
    if 'ST' not in project_name or slide_name not in gene_counts_prefixes:
        return {
            'gene_exp': '<table><tr><td>no gene data for this spot</td></tr></table>',
            'gene_dict': ''
        }

    x = int(float(params['x']))
    y = int(float(params['y']))
    gene_name_id = '{}_{}_gene_names'.format(project_name, slide_name)
    image_id = '{}_{}_x{}_y{}'.format(project_name, slide_name, x, y)
    gene_names_bytes = txn_gene.get(gene_name_id.encode('ascii'))
    gene_bytes = txn_gene.get(image_id.encode('ascii'))

    if gene_names_bytes and gene_bytes:
        gene_names = json.loads(gene_names_bytes.decode('ascii'))['gene_names']
        gene_vec = np.frombuffer(gene_bytes, dtype=np.float32).reshape(2, -1)
        inds = gene_vec[1, :].astype(np.int32)

        table_str = [
            '<table border="1"><tr><th>Gene Name</th><th>Exp value</th></tr>']
        for ind in inds:
            table_str.append(
                '<tr><td>{}</td><td>{:.6f}</td></tr>'.format(gene_names[ind], gene_vec[0, ind]))
        table_str.append('</table>')
        gene_exp = ''.join(table_str)
        gene_dict = {gene_names[ind]: '{:.6f}'.format(
            gene_vec[0, ind]) for ind in inds}
    else:
        gene_exp = '<table border="1"><tr><td>no gene data for this spot</td></tr></table>'
        gene_dict = {}
    return {'gene_exp': gene_exp, 'gene_dict': json.dumps(gene_dict)}


def compute_mean_std_cosine_similarity_from_random1000(query_embedding, search_project='ALL'):
    distances = 1 - pairwise_distances(query_embedding.reshape(1, -1),
                                       randomly_1000_data[search_project if search_project in randomly_1000_data.keys(
                                       ) else 'ALL']['embeddings'],
                                       metric='cosine')[0]
    return np.mean(distances), np.std(distances), distances


def get_image_patches(image):
    # w = 2200
    # h = 2380
    sizex, sizey = 256, 256
    w, h = image.size
    if w < sizex:
        image1 = Image.new(image.mode, (256, h), (0, 0, 0))
        image1.paste(image, ((256 - w) // 2, 0))
        image = image1
    w, h = image.size
    if h < sizey:
        image1 = Image.new(image.mode, (w, 256), (0, 0, 0))
        image1.paste(image, (0, (256 - h) // 2))
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


def get_query_embedding(img_urls, resize=0):
    image_patches_all = []
    patch_coords_all = []
    image_shown_all = []
    minWorH = 1e8
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
        if 0 < resize:
            resize_scale = 1. / 2**resize
            newW, newH = int(W*resize_scale), int(H*resize_scale)
            minWorH = min(min(newW, newH), minWorH)
            image = image.resize((newW, newH))
        patches, patch_coords, image_shown = get_image_patches(image)
        image_patches_all.append(patches)
        patch_coords_all.append(patch_coords)
        image_shown_all.append(image_shown)

    image_patches = [
        patch for patches in image_patches_all for patch in patches]
    image_urls_all = {}
    results_dict = None
    with torch.no_grad():
        images = image_processor(images=image_patches, return_tensors='pt')['pixel_values']
        feat_after_encoder_feat = feature_extractor.get_image_features(images).detach()

        # extract feat_before_attention_feat
        embedding = feat_after_encoder_feat @ state_dict['MODEL_STATE']['attention_net.0.weight'].T + \
            state_dict['MODEL_STATE']['attention_net.0.bias']
        # get the attention scores
        results_dict = attention_model(
            feat_after_encoder_feat.unsqueeze(0))
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

    embedding = embedding.reshape(1, -1)
    embedding /= np.linalg.norm(embedding)
    return embedding, image_urls_all, results_dict, minWorH


@app.route('/image_search_celery', methods=['POST', 'GET'])
def image_search_celery():

    params = request.get_json()
    if 'faiss' in params['search_method']:
        task = image_search_by_faiss_multi_celery.apply_async(args=(params,))
        return jsonify({}), 202, {'Location': url_for('image_search_celery_taskstatus', task_id=task.id)}
    return {}


@app.route('/status/<task_id>')
def image_search_celery_taskstatus(task_id):
    task = image_search_by_faiss_multi_celery.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', '')
        }
        if 'result' in task.info:
            response['result'] = task.info['result']
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),  # this is the exception raised
        }
    return jsonify(response)


@celery.task(bind=True)
def image_search_by_faiss_multi_celery(self, params):
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
        search_method = 'faiss_IndexFlatIP'
    if search_method == 'faiss_IndexFlatIP':
        search_project = 'ST'
    start = time.perf_counter()

    query_embedding, images_shown_urls, results_dict, minWorH = get_query_embedding(params['img_urls'], resize=int(float(params['resize'])))  # un-normalized
    query_embedding = query_embedding.reshape(1, -1)

    coxph_html_dict = {}

    query_embedding /= np.linalg.norm(query_embedding)  # l2norm normalized

    self.update_state(state='PROGRESS', meta={'current': 5, 'total': 100, 'status': ''})

    query_embedding_binary = None
    if search_method == 'faiss_BinaryFlat':
        query_embedding_binary = (query_embedding + 1.) * 128
        query_embedding_binary = np.clip(
            np.round(query_embedding_binary), 0, 256).astype(np.uint8)
    if 'Binary' in search_method and 'ITQ' in search_method:
        query_embedding_binary = faiss_indexes[search_method][project_names[0]]['binarizer'].sa_encode(
            query_embedding)

    self.update_state(state='PROGRESS', meta={'current': 10, 'total': 100, 'status': ''})

    D, I = knn_search_images_by_faiss(query_embedding_binary if 'Binary' in search_method else query_embedding,
                                      k=k, search_project=search_project,
                                      search_method=search_method,
                                      self=self)
    
    self.update_state(state='PROGRESS', meta={'current': 70, 'total': 100, 'status': ''})

    random1000_mean, random1000_std, random1000_dists = compute_mean_std_cosine_similarity_from_random1000(
        query_embedding, search_project=search_project)
    final_response = {}
    iinds = np.argsort(I)
    D = D[iinds].tolist()
    I = I[iinds].tolist()
    db_conn, db_cursor = get_db()
    
    if TCGA_COMBINED != '':
        hidare_table_str = f'faiss_table{TCGA_COMBINED}'
    else:
        hidare_table_str = 'faiss_table_20231120' if HIDARE_VERSION == '' else f'faiss_table{HIDARE_VERSION}'

    try:
        db_cursor.execute(
            'select * from {} where rowid in ({})'.format(hidare_table_str, ', '.join([str(ind + 1) for ind in I])))
    except:
        db_conn, db_cursor = get_db()
        db_cursor.execute(
            'select * from {} where rowid in ({})'.format(hidare_table_str, ', '.join([str(ind + 1) for ind in I])))

    res = db_cursor.fetchall()
    db_conn.close()
    infos = {int(item[0]) - 1: item for item in res}
    del res
    gc.collect()
    self.update_state(state='PROGRESS', meta={'current': 95, 'total': 100, 'status': ''})

    for score, ind in zip(D, I):

        if ind not in infos:
            continue

        rowid, x, y, svs_prefix_id, proj_id = infos[ind]

        item = {'_score': score,
                '_zscore': (score - random1000_mean) / random1000_std,
                '_pvalue': len(np.where(random1000_dists >= score)[0]) / len(random1000_dists)}
        project_name = list(project_names_dict.keys())[proj_id]
        slide_name = list(svs_prefixes_dict.keys())[svs_prefix_id]
        x0, y0 = int(x), int(y)
        image_id = '{}_{}_x{}_y{}'.format(project_name, slide_name, x, y)
        image_name = '{}_x{}_y{}'.format(slide_name, x, y)

        if 'ST' in project_name:
            gene_bytes = txn_gene.get(image_id.encode('ascii'))
            has_gene = '1' if gene_bytes else '0'
        else:
            has_gene = '0'

        image_id_bytes = image_id.encode('ascii')
        img_bytes = None
        for i in range(len(txns[project_name])):
            if img_bytes is None:
                img_bytes = txns[project_name][i].get(image_id_bytes)

        if img_bytes is None:
            continue

        im = Image.open(io.BytesIO(img_bytes))
        buffer = io.BytesIO()
        im.save(buffer, format="jpeg")
        encoded_image = base64.b64encode(buffer.getvalue()).decode()
        img_url = "data:image/jpeg;base64, " + encoded_image

        key = '{}_{}'.format(project_name, slide_name)
        if key in all_scales:
            scale = all_scales[key]['scale']
            patch_size_vis_level = all_scales[key]['patch_size_vis_level']
        else:
            scale = [1.0, 1.0]
            patch_size_vis_level = 256

        x = int(scale[0] * float(image_name.split('_')[-2].replace('x', '')))
        y = int(scale[1] * float(image_name.split('_')[-1].replace('y', '')))

        if slide_name in all_notes:
            note = all_notes[slide_name]
        else:
            note = 'No clinical information.'
        note_generated = ''

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
            final_response[slide_name]['comment'] = note
            final_response[slide_name]['comment_generated'] = note_generated
        else:
            final_response[slide_name] = {}
            final_response[slide_name]['project_name'] = project_name if project_name != "KenData" else "NCIData"
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
            final_response[slide_name]['comment'] = note
            final_response[slide_name]['comment_generated'] = note_generated
    end = time.perf_counter()
    time_elapsed_ms = (end - start) * 1000

    zscore_sum_list = []
    for k in final_response.keys():
        final_response[k]['min_score'] = float(min(final_response[k]['scores']))
        final_response[k]['max_score'] = float(max(final_response[k]['scores']))
        final_response[k]['min_zscore'] = float(min(final_response[k]['zscores']))
        final_response[k]['max_zscore'] = float(max(final_response[k]['zscores']))
        zscore_sum = float(sum(final_response[k]['zscores']))
        final_response[k]['zscore_sum'] = zscore_sum
        zscore_sum_list.append(abs(zscore_sum))
        final_response[k]['_random1000_mean'] = float(random1000_mean)
        final_response[k]['_random1000_std'] = float(random1000_std)
        final_response[k]['_time_elapsed_ms'] = float(time_elapsed_ms)
    sort_inds = np.argsort(zscore_sum_list)[::-1].tolist()
    allkeys = list(final_response.keys())
    ranks = {rank: allkeys[ind] for rank, ind in enumerate(sort_inds)} # sorted by zscore_sum descend order

    # prediction
    table_str = [
        '<table border="1"><tr><th>task</th><th>prediction</th></tr>']
    for k,v in argsdata['classification_dict'].items():
        Y_prob_k = F.softmax(results_dict[k + '_logits'], dim=1).detach().numpy()[0]
        table_str.append(
            '<tr><td>{}</td><td>{}: {:.3f}</td></tr>'.format(k.replace('_cls', ''), v[1], Y_prob_k[1]))
    for k in argsdata['regression_list']:
        table_str.append(
            '<tr><td>{}</td><td>{:.3f}</td></tr>'.format(k, results_dict[k + '_logits'].item()))
    table_str.append('</table>')
    pred_str = ''.join(table_str)

    gc.collect()
    self.update_state(state='PROGRESS', meta={'current': 99, 'total': 100, 'status': ''})
    return {'current': 100, 'total': 100, 'status': 'Task completed!',
            'result': {'coxph_html_dict': coxph_html_dict, 'response': final_response, 'ranks': ranks, 'pred_str': pred_str,
            'images_shown_urls': images_shown_urls, 'minWorH': minWorH}}


app.wsgi_app = ProxyFix(app.wsgi_app)






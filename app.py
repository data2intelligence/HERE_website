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

# (CHANGE THIS)
DB_USER = "root"
DB_PASSWORD = "JiangLab_NCI_123"
DB_HOST = "localhost"
DB_DATABASE = "hidare_app"  


DATA_DIR = f'data_HiDARE_PLIP_20240208'
ST_ROOT = f'{DATA_DIR}/20240202v4_ST/PanCancer2GPUsFP/shared_attention_imagenetPLIP/split1_e95_h224_density_vis/feat_before_attention_feat/test/analysis/one_patient_top_128'
project_names = ['TCGA-COMBINED', 'KenData', 'ST']  # do not change the order
project_start_ids = {'TCGA-COMBINED': 0, 'KenData': 159402517, 'ST': 239029686}

def get_db():
    db_conn = pymysql.connect(user=DB_USER, password=DB_PASSWORD, host=DB_HOST, database=DB_DATABASE)
    db_conn.autocommit = False
    db_cursor = db_conn.cursor()
    return db_conn, db_cursor

feature_extractor = None
config = None
transform = None
state_dict = None

attention_model = AttentionModel()
feature_extractor = CLIPModel.from_pretrained('vinid/plip')
image_processor = CLIPProcessor.from_pretrained('vinid/plip')
feature_extractor = feature_extractor.eval()

# encoder + attention except final
state_dict = torch.load(f"{DATA_DIR}/assets/snapshot_95.pt", map_location='cpu')
attention_model.load_state_dict(state_dict['MODEL_STATE'], strict=False)
attention_model = attention_model.eval()

faiss_Ms = [32]
faiss_nlists = [128]
faiss_indexes = {'faiss_IndexFlatIP': {}}
for project_name in ['ST']:
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

with open(f'{DATA_DIR}/assets/randomly_1000_data.pkl', 'rb') as fp:
    randomly_1000_data = pickle.load(fp)

font = ImageFont.truetype("Gidole-Regular.ttf", size=36)

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

app = Flask(__name__,
            static_url_path='',
            static_folder='',
            template_folder='')

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


def knn_search_images_by_faiss(query_embedding, k=10, search_project="ALL", search_method='faiss'):
    if search_project == 'ALL':
        Ds, Is = {}, {}
        for iiiii, project_name in enumerate(project_names):
            if search_method == 'faiss_IndexFlatIP':
                Di, Ii = faiss_indexes['faiss_IndexFlatIP'][project_name].search(
                    query_embedding, k)
            else: # 'HNSW' in search_method:
                Di, Ii = faiss_indexes[search_method][project_name].search(
                    query_embedding, k)
            Ii_filtered = [ii for ii in Ii[0] if ii>=0]
            Di = np.array([dd for dd, ii in zip(Di[0], Ii[0]) if ii>=0])
            beginid = project_start_ids[project_name]
            Ii = [beginid+ii for ii in Ii[0] if ii>=0]
            Ds[project_name] = Di
            Is[project_name] = Ii

        D = np.concatenate(list(Ds.values()))
        I = np.concatenate(list(Is.values()))
        if 'HNSW' in search_method:
            inds = np.argsort(D)[:k]
        else:  # IP or cosine similarity, the larger, the better
            inds = np.argsort(D)[::-1][:k]
        return D[inds], I[inds]

    else:
        if search_method == 'faiss_IndexFlatIP':
            Di, Ii = faiss_indexes['faiss_IndexFlatIP'][search_project].search(
                query_embedding, k)
        else: # using 'HNSW':
            Di, Ii = faiss_indexes[search_method][search_project].search(
                query_embedding, k)

        Ii_filtered = [ii for ii in Ii[0] if ii>=0]
        Di = np.array([dd for dd, ii in zip(Di[0], Ii[0]) if ii>=0])
        beginid = project_start_ids[search_project]
        Ii = np.array([beginid+ii for ii in Ii[0] if ii>=0])
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

@app.route('/download_gene2', methods=['POST', 'GET'])
def download_gene2():
    items = request.get_json()

    fh = io.BytesIO()
    tar_fp = tarfile.open(fileobj=fh, mode='w:gz')

    good = False
    for item in items:
        project_name = item['project_name']
        slide_name = item['slide_name']
        try:
            gene_names = [vv.strip().upper() for vv in item['gene_names'].split(',') if len(vv) > 0]
        except:
            gene_names = []
        vst_filename = f'{ST_ROOT}/../../vst_dir_db/{slide_name}_original_VST.db'
        coords = np.array(item['coords'].split(',')[:-1]).astype('uint16').reshape(-1,2)
        try:
            df = pd.read_parquet(vst_filename, filters=[[('__upperleft_X', '=', x), ('__upperleft_Y', '=', y)] for x, y in coords])
        except:
            df = None
        if df is None:
            continue
        if len(gene_names) == 0:
            df = df.drop(columns=['__upperleft_X', '__upperleft_Y'])
        else:
            df = df[gene_names+['__spot_X', '__spot_Y']]
        if len(df.columns) == 0:
            continue
        b_buf = io.BytesIO(df.to_csv().encode())
        info = tarfile.TarInfo(name="{}_{}_selected_spots.csv".format(project_name, slide_name)) 
        info.size = b_buf.getbuffer().nbytes
        info.mtime = time.time()
        b_buf.seek(0)
        tar_fp.addfile(info, b_buf)
        good = True
    tar_fp.close()
    if good:
        filepath = '{}/temp_images_dir/selected_genes_{}.tar.gz'.format(DATA_DIR, time.time()) 
        with open(filepath, 'wb') as fp:
            fp.write(fh.getvalue())
        return {'filepath': filepath}
    else:
        return {'filepath': ''}


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


@app.route('/gene_search', methods=['POST', 'GET'])
def gene_search():
    params = request.get_json()
    start = time.perf_counter()

    gene_names = []
    cohensd_thres = None
    try:
        gene_names = [vv for vv in params['gene_names'].split(',') if len(vv) > 0]
        cohensd_thres = float(params['cohensd_thres'])
    except:
        pass

    status = ''
    if len(gene_names) == 0: 
        status = 'Invalid input gene names. \n'
    if cohensd_thres == None:
        cohensd_thres = 1.0
        status += 'Cohensd values should be float. Set it to 1.0. \n'

    db_conn, db_cursor = get_db()
    hidare_table_str = f'cluster_result_table_20240208 as a, gene_table_20240208 as b, '\
        f'cluster_table_20240208 as c, cluster_setting_table_20240208 as d, st_table_20240208 as e'

    if len(gene_names) == 0:
        if cohensd_thres is None:
            sql = f'select a.*, b.symbol, d.cluster_setting, e.prefix, c.cluster_label, c.cluster_info, f.note from {hidare_table_str} '\
                'where a.gene_id = b.id and c.id = a.c_id and c.cs_id = d.id and '\
                    'c.st_id = e.id order by a.cohensd desc limit 100;'
            print('sql1', sql, gene_names, cohensd_thres)
            try:
                db_cursor.execute(sql)
            except:
                db_conn, db_cursor = get_db()
                db_cursor.execute(sql)

        else:
            sql = f'select a.*, b.symbol, d.cluster_setting, e.prefix, c.cluster_label, c.cluster_info from {hidare_table_str} '\
                'where a.cohensd > %s and a.gene_id = b.id and c.id = a.c_id and c.cs_id = d.id '\
                    'and c.st_id = e.id order by a.cohensd desc limit 100;'
            print('sql2', sql, gene_names, cohensd_thres)
            try:
                db_cursor.execute(sql, (cohensd_thres,))
            except:
                db_conn, db_cursor = get_db()
                db_cursor.execute(sql, (cohensd_thres,))

    else:
        if cohensd_thres is None:
            sql = f'select a.*, b.symbol, d.cluster_setting, e.prefix, c.cluster_label, c.cluster_info from {hidare_table_str} '\
                'where b.symbol in %s and a.gene_id = b.id and c.id = a.c_id and c.cs_id = d.id '\
                    'and c.st_id = e.id order by a.cohensd desc limit 100;'
            print('sql3', sql, gene_names, cohensd_thres)
            try:
                db_cursor.execute(sql, (gene_names,))
            except:
                db_conn, db_cursor = get_db()
                db_cursor.execute(sql, (gene_names,))
        else:
            sql = f'select a.*, b.symbol, d.cluster_setting, e.prefix, c.cluster_label, c.cluster_info from {hidare_table_str} '\
                'where b.symbol in %s and a.cohensd > %s and a.gene_id = b.id and c.id = a.c_id and '\
                    'c.cs_id = d.id and c.st_id = e.id order by a.cohensd desc limit 100;'
            print('sql4', sql, gene_names, cohensd_thres)
            try:
                db_cursor.execute(sql, (gene_names, cohensd_thres))
            except:
                db_conn, db_cursor = get_db()
                db_cursor.execute(sql, (gene_names, cohensd_thres))

    result = db_cursor.fetchall()
    if result is None or len(result) == 0:
        return {'current': 100, 'total': 100, 'status': 'Task completed!',
            'result': {'coxph_html_dict': '', 'response': '', 'images_shown_urls': ''}}

    result_df = pd.DataFrame(result, columns=['id', 'c_id', 'gene_id', 'cohensd', 'pvalue', 'pvalue_corrected',
                             'zscore', 'gene_symbol', 'cluster_setting', 'ST_prefix', 'cluster_label', 'cluster_info'])
    removed_columns = ['id', 'c_id', 'gene_id']
    result_df = result_df.drop(columns=removed_columns)

    sql = f'select note from image_table_20240628 where svs_prefix in %s'
    ST_prefixes = result_df['ST_prefix'].value_counts().index.values.tolist()
    try:
        db_cursor.execute(sql, (ST_prefixes, ))
    except:
        db_conn, db_cursor = get_db()
        db_cursor.execute(sql, (gene_names, cohensd_thres))
    result = db_cursor.fetchall()
    ST_notes = {prefix: note[0] for prefix, note in zip(ST_prefixes, result)}
    db_conn.close()

    root = ST_ROOT
    notes = []
    patch_annotations = []
    for rowid, row in result_df.iterrows():
        prefix = row['ST_prefix']
        cluster_setting = row['cluster_setting']
        cluster_label = int(float(row['cluster_label']))
        with open(os.path.join(root, cluster_setting, prefix, prefix+'_cluster_data.pkl'), 'rb') as fp:
            cluster_labels = pickle.load(fp)['cluster_labels']
        with open(os.path.join(root, cluster_setting, prefix, prefix+'_patches_annotations.json'), 'r') as fp:
            all_patch_annotations = json.load(fp)
        inds = np.where(cluster_labels == cluster_label)[0]
        patch_annotations.append([all_patch_annotations[indd] for indd in inds])

        if prefix in ST_notes:
            notes.append(ST_notes[prefix])
        else:
            notes.append('No clinical information.\n')
    result_df['note'] = notes
    result_df['annotations'] = patch_annotations

    gc.collect()

    return {'current': 100, 'total': 100, 'status': 'Task completed!',
            'result': {'coxph_html_dict': '', 'response': json.dumps(result_df.to_dict()), 'images_shown_urls': ''}}



@app.route('/get_gene_map', methods=['POST', 'GET'])
def get_gene_map():
    items = request.get_json()
    gene_names = [n.strip().upper() for n in items['gene_names'].split(',')]
    slide_name = items['slide_name']
    results_path = items['results_path']
    vst_filename = f'{results_path}/../../../../vst_dir_db/{slide_name}.db'
    if not os.path.exists(vst_filename):
        return {'current': 100, 'total': 100, 'status': 'Task completed!',
            'result': {}}

    parquet_file = pq.ParquetFile(vst_filename)
    existing_columns = parquet_file.schema.names

    valid_gene_names = [v for v in gene_names if v in existing_columns]
    if len(valid_gene_names) == 0:
        return {'current': 100, 'total': 100, 'status': 'Task completed!',
            'result': {}}

    query_columns = ['__coordX', '__coordY', '__circle_radius'] + valid_gene_names
    df = pd.read_parquet(vst_filename, columns=query_columns)

    cmap = plt.get_cmap('jet')
    def mapper(v):
        return '%02x%02x%02x' % tuple([int(x * 255) for x in cmap(v)[:3]])
    df[valid_gene_names] = df[valid_gene_names].applymap(mapper)

    response_dict = {'status': 'alldone', 'results': {}}
    for gene_name in valid_gene_names:
        response_dict['results'][gene_name] = '\n'.join([
            '{},{},{},{},{}'.format(
                ind, row['__coordX'], row['__coordY'], row['__circle_radius'], row[gene_name].upper()
            )
            for ind, row in df.iterrows()
        ])

    # return response_dict
    return {'current': 100, 'total': 100, 'status': 'Task completed!',
            'result': response_dict}


def compute_mean_std_cosine_similarity_from_random1000(query_embedding, search_project='ALL'):
    distances = 1 - pairwise_distances(query_embedding.reshape(1, -1),
                                       randomly_1000_data[search_project if search_project in randomly_1000_data.keys() else 'ALL'],
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


@app.route('/image_search', methods=['POST', 'GET'])
def image_search():
    params = request.get_json()
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

    query_embedding, images_shown_urls, results_dict, minWorH = \
        get_query_embedding(params['img_urls'], resize=int(float(params['resize'])))  # un-normalized
    query_embedding = query_embedding.reshape(1, -1)

    coxph_html_dict = {}

    query_embedding /= np.linalg.norm(query_embedding)  # l2norm normalized

    D, I = knn_search_images_by_faiss(query_embedding,
                                      k=k, search_project=search_project,
                                      search_method=search_method)
    
    random1000_mean, random1000_std, random1000_dists = compute_mean_std_cosine_similarity_from_random1000(
        query_embedding, search_project=search_project)
    final_response = {}
    iinds = np.argsort(I)
    D = D[iinds].tolist()
    I = I[iinds].tolist()
    db_conn, db_cursor = get_db()
    
    hidare_table_str = f'faiss_table_20240619 as a, image_table_20240628 as b'
    try:
        db_cursor.execute(
            'select a.*, b.scale, b.patch_size_vis_level, b.svs_prefix, b.note from {} where '\
                'a.rowid in ({}) and a.svs_prefix_id = b.svs_prefix_id and '\
                    'a.project_id = b.project_id'.format(hidare_table_str, ', '.join([str(ind + 1) for ind in I])))
    except:
        db_conn, db_cursor = get_db()
        db_cursor.execute(
            'select a.*, b.scale, b.patch_size_vis_level, b.svs_prefix, b.note from {} where '\
                'a.rowid in ({}) and a.svs_prefix_id = b.svs_prefix_id and '\
                    'a.project_id = b.project_id'.format(hidare_table_str, ', '.join([str(ind + 1) for ind in I])))

    res = db_cursor.fetchall()
    db_conn.close()
    infos = {int(item[0]) - 1: item for item in res}
    del res
    gc.collect()

    for score, ind in zip(D, I):

        if ind not in infos:
            continue

        rowid, x, y, svs_prefix_id, proj_id, scale, patch_size_vis_level, slide_name, note = infos[ind]
        scale = float(scale)
        patch_size_vis_level = int(patch_size_vis_level)
        if len(note) == 0:
            note = 'No clinical information. '

        item = {'_score': score,
                '_zscore': (score - random1000_mean) / random1000_std,
                '_pvalue': len(np.where(random1000_dists >= score)[0]) / len(random1000_dists)}
        project_name = project_names[proj_id]
        x0, y0 = int(x), int(y)
        image_id = '{}_{}_x{}_y{}'.format(project_name, slide_name, x, y)
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
            final_response[slide_name]['note'] = note
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
    for k,v in attention_model.classification_dict.items():
        Y_prob_k = F.softmax(results_dict[k + '_logits'], dim=1).detach().numpy()[0]
        table_str.append(
            '<tr><td>{}</td><td>{}: {:.3f}</td></tr>'.format(k.replace('_cls', ''), v[1], Y_prob_k[1]))
    for k in attention_model.regression_list:
        table_str.append(
            '<tr><td>{}</td><td>{:.3f}</td></tr>'.format(k, results_dict[k + '_logits'].item()))
    table_str.append('</table>')
    pred_str = ''.join(table_str)

    gc.collect()
    return {'current': 100, 'total': 100, 'status': 'Task completed!',
            'result': {'coxph_html_dict': coxph_html_dict, 'response': final_response, 'ranks': ranks, 'pred_str': pred_str,
            'images_shown_urls': images_shown_urls, 'minWorH': minWorH}}


app.wsgi_app = ProxyFix(app.wsgi_app)

from app_common import *



def main_jiang_exp1():
    files = sorted(glob.glob('/mnt/hidare-efs/data_20240208/jiang_exp1/png/*.png'))
    print('files', files)
    search_backbone = 'HERE_ProvGigaPath'
    search_backbone = 'HERE_CONCH'
    save_dir = f'/mnt/hidare-efs/data_20240208/jiang_exp1/search_results_{search_backbone}'
    os.makedirs(save_dir, exist_ok=True)
    for ind, f in enumerate(files):
        print('begin ', f)
        prefix = os.path.basename(f).replace('.png', '')
        save_filename = os.path.join(save_dir, f'{prefix}.pkl')
        if os.path.exists(save_filename):
            continue
        params = {
            'k': 50,
            'search_project': 'ST',
            'search_feature': 'before_attention',
            'search_method': 'faiss_IndexHNSWFlat_m32_IVFPQ_nlist128_m8',
            'socketid': '',
            'img_urls': [f],
            'filenames': [f],
            'resize': 0,
            'search_backbone': search_backbone
        }
        result = image_search_main(params=params)

        with open(save_filename, 'wb') as fp:
            pickle.dump(result, fp)

        # if ind==2:
        #     break



def main_jiang_step2():
    import sys,os,glob
    import openslide
    import pickle
    from PIL import Image
    import numpy as np
    import cv2
    import pandas as pd
    from PIL import Image, ImageDraw
    import h5py

    prefix = 'HERE_'
    backbone = 'ProvGigaPath'
    backbone = 'CONCH'
    save_root = f'/data/Jiang_Lab/Data/Zisha_Zhong/jiang_exp1/search_results_{prefix}{backbone}'
    svs_dir = '/data/zhongz2/ST_256/svs'
    patches_dir = '/data/zhongz2/ST_256/patches'
    result_dir = '/data/Jiang_Lab/Data/Zisha_Zhong/temp_20240202/differential_results/20240202v4_ST/PanCancer2GPUsFP/shared_attention_imagenetPLIP/split1_e95_h224_density_vis/feat_before_attention_feat/test'
    result_dir = '/data/zhongz2/temp29/debug/debug_results/ngpus2_accum4_backboneProvGigaPath_dropout0.25/analysis/ST/ProvGigaPath/feat_before_attention_feat'
    
    result_dir = '/data/zhongz2/temp29/debug/results_20240724_e100/ngpus2_accum4_backboneCONCH_dropout0.25/analysis/ST/CONCH/feat_before_attention_feat/'

    files = sorted(glob.glob(os.path.join(save_root, '*.pkl')))
    topk = 5

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 2
    fontColor              = (255,255,255)
    thickness              = 2
    lineType               = 2

    df = pd.read_excel('/data/Jiang_Lab/Data/Zisha_Zhong/temp_20240208_hidare/ST_list_cancer1.xlsx', index_col='ID')

    query_prefixes = [os.path.basename(f).replace('.pkl', '') for f in files]
    all_results = []
    for prefix in query_prefixes:

        f = os.path.join(save_root, prefix+'.pkl')
        if not os.path.exists(f):
            continue
        query_prefix = os.path.basename(f).replace('.pkl', '')
        print('begin ', query_prefix)
        with open(f, 'rb') as fp:
            result = pickle.load(fp)['result']
        ranks = result['ranks']
        response = result['response']
        expand = 3

        for k in range(topk):
            if k not in ranks:
                all_results.append(
                ('', '', '', '', '', '',
                    '','',
                    '')
                )
                continue
            svs_prefix = ranks[k]

            with h5py.File(os.path.join(patches_dir, svs_prefix + '.h5'), 'r') as file:
                patch_size0 = file['coords'].attrs['patch_size']
                patch_level0 = file['coords'].attrs['patch_level']

            gene_data_filename = f'{result_dir}/gene_data/{svs_prefix}_gene_data.pkl'
            cluster_data_filename = f'{result_dir}/analysis/one_patient_top_128/meanstd_none_kmeans_euclidean_8_1_clustering/{svs_prefix}/{svs_prefix}_cluster_data.pkl'
            vst_filename = f'{result_dir}/vst_dir/{svs_prefix}.tsv'
            if not os.path.exists(gene_data_filename) or not os.path.exists(cluster_data_filename) or not os.path.exists(vst_filename):
                print(svs_prefix, 'not existed')
                continue

            with open(gene_data_filename, 'rb') as fp:
                gene_data_dict = pickle.load(fp)
            with open(cluster_data_filename, 'rb') as fp:
                cluster_data = pickle.load(fp)
            barcode_col_name = gene_data_dict['barcode_col_name']
            Y_col_name = gene_data_dict['Y_col_name']
            X_col_name = gene_data_dict['X_col_name']
            mpp = gene_data_dict['mpp']
            coord_df = gene_data_dict['coord_df']
            counts_df = gene_data_dict['counts_df']

            vst = pd.read_csv(vst_filename, sep='\t', index_col=0)
            vst = vst.subtract(vst.mean(axis=1), axis=0)

            barcodes = coord_df[barcode_col_name].values.tolist()
            stY = coord_df[Y_col_name].values.tolist()
            stX = coord_df[X_col_name].values.tolist()

            st_patch_size = int(
                pow(2, np.ceil(np.log(64 / mpp) / np.log(2))))
            st_all_coords = np.array([stX, stY]).T
            st_all_coords[:, 0] -= st_patch_size // 2
            st_all_coords[:, 1] -= st_patch_size // 2
            st_all_coords = st_all_coords.astype(np.int32)

            vst = vst.T
            vst.index.name = 'barcode'
            valid_barcodes = set(vst.index.values.tolist())
            # print(len(valid_barcodes))

            # cluster_data_dict['coords_in_original']
            cluster_coords = cluster_data['coords_in_original']
            # cluster_data_dict['cluster_labels']
            cluster_labels = cluster_data['cluster_labels']
            # cluster_coords = cluster_data_dict['coords_in_original']
            # cluster_labels = cluster_data_dict['cluster_labels']

            cluster_barcodes = []
            innnvalid = 0
            iinds = []
            for iiii, (x, y) in enumerate(cluster_coords):
                ind = np.where((st_all_coords[:, 0] == x) & (
                    st_all_coords[:, 1] == y))[0]
                if len(ind) > 0:
                    barcoode = barcodes[ind[0]]
                    if barcoode in valid_barcodes:
                        cluster_barcodes.append(barcoode)
                        iinds.append(iiii)
                else:
                    innnvalid += 1
            cluster_labels = cluster_labels[iinds]
            cluster_coords = cluster_coords[iinds]
            vst1 = vst.loc[cluster_barcodes]
            counts_df1 = counts_df.T
            coord_df1 = coord_df.set_index(barcode_col_name)
            coord_df1.index.name = 'barcode'
            coord_df1 = coord_df1.loc[cluster_barcodes]
            stY = coord_df1[Y_col_name].values.tolist()
            stX = coord_df1[X_col_name].values.tolist()
            final_df = coord_df1.copy()
            
            save_dir = os.path.join(save_root, 'hidare_results', query_prefix, f'top-{k}-{svs_prefix}')
            os.makedirs(save_dir, exist_ok=True)

            slide = openslide.open_slide(os.path.join(svs_dir, svs_prefix+'.svs'))
            spot_size = df.loc[svs_prefix, 'spot_size']
            st_patch_size = df.loc[svs_prefix, 'patch_size']
            circle_radius = int(spot_size * 0.5)
            W0, H0 = slide.level_dimensions[0]
            img = slide.read_region((0, 0), 0, (W0, H0)).convert('RGB')
            draw = ImageDraw.Draw(img)
            for ind, (x,y) in enumerate(zip(stX, stY)):
                xy = [x-circle_radius, y-circle_radius, x+circle_radius, y+circle_radius]
                draw.ellipse(xy, outline=(255, 128, 0), width=8)

            img2 = Image.fromarray(255*np.ones((H0, W0, 3), dtype=np.uint8))
            draw2 = ImageDraw.Draw(img2)

            item = response[svs_prefix]
            min_score, max_score = item['min_score'], item['max_score']
            min_zscore, max_zscore = item['min_zscore'], item['max_zscore']
            zscore_sum = item['zscore_sum']
            scores = item['scores']
            zscores = item['zscores']

            all_results.append(
                (query_prefix, k, svs_prefix, len(scores), min_score, max_score,
                min_zscore,max_zscore,
                zscore_sum)
            )
            for pi, patch in enumerate(item['images']):
                # x0, y0, patch_level0, patch_size0 = \
                #     patch['x0'], patch['y0'], patch['patch_level0'], patch['patch_size0']
                x0, y0 = patch['x0'], patch['y0']
                xy = [x0, y0, x0+patch_size0, y0+patch_size0]
                draw2.rectangle(xy, fill=(0,255,0), width=5)

                x, y = patch['x'], patch['y']
                scale = x / x0
                W,H = slide.level_dimensions[patch_level0]
                x1 = max(0, x0 - expand*patch_size0)
                y1 = max(0, y0 - expand*patch_size0)
                x2, y2 = min(W-1,x1+(2*expand+1)*patch_size0), min(H-1,y1+(2*expand+1)*patch_size0)
                x3, y3 = x0 - x1, y0 - y1  # patch_size0

                im = np.array(slide.read_region(location=(x1, y1), level=patch_level0, size=(x2-x1, y2-y1)).convert('RGB'))
                cv2.rectangle(im, (x3, y3), (x3+patch_size0, y3+patch_size0),(255,255,255),2)

                cv2.putText(im,'dist: {:.3f}, zscore: {:.3f}'.format(scores[pi], zscores[pi]), 
                    (x3, y3-5), 
                    font, 
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)

                cv2.imwrite(os.path.join(save_dir, 'patch-{:05d}.jpg'.format(pi)), im[:,:,::-1])  # RGB --> BGR
                print(pi, ' done')

            img3 = Image.blend(img, img2, alpha=0.4)
            img3.save(os.path.join(save_dir, f'{svs_prefix}.jpg'))
            del img, img2, img3


        all_results.append(
            ('', '', '', '', '', '',
                '','',
                '')
        )
    df = pd.DataFrame(all_results, columns=['query', 'rank', 'match_filename', 'num_patches', 'min_distance', 'max_distance', 'min_zscore', 'max_zscore', 'zscore_sum'])
    df.to_excel(os.path.join(save_root, 'hidare_result.xlsx'))




# 20240715
def main_here101_with_r2r4():
    files = sorted(glob.glob('/mnt/hidare-efs/data_20240208/allpng_with_r2r4/png/*.png'))
    print('files', files)
    search_backbone = 'HERE_ProvGigaPath'
    save_dir = f'/mnt/hidare-efs/data_20240208/allpng_with_r2r4/search_results_{search_backbone}'
    os.makedirs(save_dir, exist_ok=True)
    for ind, f in enumerate(files):
        print('begin ', f)
        prefix = os.path.basename(f).replace('.png', '')
        save_filename = os.path.join(save_dir, f'{prefix}.pkl')
        if os.path.exists(save_filename):
            continue
        params = {
            'k': 50,
            'search_project': 'KenData',
            'search_feature': 'before_attention',
            'search_method': 'faiss_IndexHNSWFlat_m32_IVFPQ_nlist128_m8',
            'socketid': '',
            'img_urls': [f],
            'filenames': [f],
            'resize': 0,
            'search_backbone': search_backbone
        }
        result = image_search_main(params=params)

        with open(save_filename, 'wb') as fp:
            pickle.dump(result, fp)

        # if ind==2:
        #     break





def main_here101_step2():

    import sys,os,glob
    import openslide
    import pickle
    from PIL import Image
    import numpy as np
    import cv2
    import pandas as pd
    import h5py

    search_backbone = 'HERE_ProvGigaPath'
    search_backbone = 'HERE_CONCH'
    postfix = ''
    postfix = '_r2r4'

    # for search_backbone in ['HERE_PLIP', 'HERE_ProvGigaPath', 'HERE_CONCH']:
    for search_backbone in ['HERE_CONCH']:
        save_root = '/data/Jiang_Lab/Data/Zisha_Zhong/jinlin35_hard/search_results'
        save_root = f'/data/Jiang_Lab/Data/Zisha_Zhong/HERE101/allpng_with_r2r4/search_results_{search_backbone}'
        save_root = f'/data/Jiang_Lab/Data/Zisha_Zhong/HERE101/allpng_with_r2r4/evaluation_6cases_20240806/search_results_{search_backbone}'
        save_root = f'/data/Jiang_Lab/Data/Zisha_Zhong/HERE101/allpng_with_r2r4/search_results_{search_backbone}'
        save_root = f'/data/Jiang_Lab/Data/Zisha_Zhong/HERE101/allpng_with_r2r4/search_results_{search_backbone}_20250403'
        images_dir = f'/data/Jiang_Lab/Data/Zisha_Zhong/HERE101/allpng_with_r2r4/png'

        save_root = f'/data/Jiang_Lab/Data/Zisha_Zhong/HERE101/allpng_with_r2r4_20250403/search_results_{search_backbone}{postfix}'
        images_dir = f'/data/Jiang_Lab/Data/Zisha_Zhong/HERE101/allpng_with_r2r4_20250403/png{postfix}'

        svs_dir = '/data/zhongz2/KenData_20240814_256/svs'
        patches_dir = '/data/zhongz2/KenData_20240814_256/patches'
        files = sorted(glob.glob(os.path.join(save_root, '*.pkl')))
        topk = 5

        font                   = cv2.FONT_HERSHEY_SIMPLEX
        fontScale              = 2
        fontColor              = (255,255,255)
        thickness              = 2
        lineType               = 2

        save_dir1 = os.path.join(save_root, 'hidare_results') # , f'top-{k}-{svs_prefix}')
        os.makedirs(save_dir1, exist_ok=True)
        query_prefixes = [os.path.basename(f).replace('.pkl', '') for f in files]
        all_results = []
        for prefix in query_prefixes:
            # if '_r2' in prefix or '_r4' in prefix:
            #     continue

            f = os.path.join(save_root, prefix+'.pkl')
            if not os.path.exists(f):
                continue

            query_prefix = os.path.basename(f).replace('.pkl', '')
            save_filename = os.path.join(save_dir1, f'{query_prefix}.jpg')
            if os.path.exists(save_filename):
                continue

            print('begin ', query_prefix)
            with open(f, 'rb') as fp:
                result = pickle.load(fp)['result']
            ranks = result['ranks']
            response = result['response']
            expand = 3

            query_img = cv2.imread(os.path.join(images_dir, prefix+'.png'))
            H, W = query_img.shape[:2]
            maxsize = max(W,H)
            if '_r2' in query_prefix or '_r4' in query_prefix:
                maxsize = max(1020, maxsize)
            pad_w = maxsize-W
            pad_h = maxsize-H
            query_img1 = np.pad(query_img, ((0, pad_h),(0,pad_w),(0,0)), constant_values=255)
            pad_img = 255*np.ones((maxsize,5,3),dtype=np.uint8)

            res_imgs = [query_img1, pad_img]
            for k in range(topk):
                if k not in ranks:
                    all_results.append(
                    ('', '', '', '', '', '',
                        '','',
                        '')
                    )
                    continue
                svs_prefix = ranks[k]

                with h5py.File(os.path.join(patches_dir, svs_prefix + '.h5'), 'r') as file:
                    patch_size0 = file['coords'].attrs['patch_size']
                    patch_level0 = file['coords'].attrs['patch_level']

                # save_dir = os.path.join(save_root, 'hidare_results', query_prefix, f'top-{k}-{svs_prefix}')
                # os.makedirs(save_dir, exist_ok=True)

                slide = openslide.open_slide(os.path.join(svs_dir, svs_prefix+'.svs'))

                item = response[svs_prefix]
                min_score, max_score = item['min_score'], item['max_score']
                min_zscore, max_zscore = item['min_zscore'], item['max_zscore']
                zscore_sum = item['zscore_sum']
                scores = item['scores']
                zscores = item['zscores']

                all_results.append(
                    (query_prefix, k, svs_prefix, len(scores), min_score, max_score,
                    min_zscore,max_zscore,
                    zscore_sum)
                )
                for pi, patch in enumerate(item['images']):
                    # x0, y0, patch_level0, patch_size0 = \
                    #     patch['x0'], patch['y0'], patch['patch_level0'], patch['patch_size0']
                    x0, y0 = patch['x0'], patch['y0']
                    x, y = patch['x'], patch['y']
                    scale = x / x0
                    W,H = slide.level_dimensions[patch_level0]
                    x1 = max(0, x0 - expand*patch_size0)
                    y1 = max(0, y0 - expand*patch_size0)
                    x2, y2 = min(W-1,x1+(2*expand+1)*patch_size0), min(H-1,y1+(2*expand+1)*patch_size0)
                    x3, y3 = x0 - x1, y0 - y1  # patch_size0

                    im = np.array(slide.read_region(location=(x1, y1), level=patch_level0, size=(x2-x1, y2-y1)).convert('RGB'))
                    cv2.rectangle(im, (x3, y3), (x3+patch_size0, y3+patch_size0),(255,255,255),2)

                    cv2.putText(im,'dist: {:.3f}, zscore: {:.3f}'.format(scores[pi], zscores[pi]), 
                        (x3, y3-5), 
                        font, 
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)

                    # cv2.imwrite(os.path.join(save_dir, 'patch-{:05d}.jpg'.format(pi)), im[:,:,::-1])  # RGB --> BGR
                    print(pi, ' done')

                    h,w = im.shape[:2]
                    cx,cy = w//2,h//2
                    res_imgs.append(im[int(cy-maxsize//2):int(cy+maxsize//2),int(cx-maxsize//2):int(cx+maxsize//2),::-1])
                    res_imgs.append(pad_img)
                    break # only one image
            
            res_imgs = np.concatenate(res_imgs, axis=1)
            cv2.imwrite(save_filename, res_imgs)  # RGB --> BGR
            all_results.append(
                ('', '', '', '', '', '',
                    '','',
                    '')
            )


        df = pd.DataFrame(all_results, columns=['query', 'rank', 'match_filename', 'num_patches', 'min_distance', 'max_distance', 'min_zscore', 'max_zscore', 'zscore_sum'])
        df.to_excel(os.path.join(save_root, 'hidare_result.xlsx'))


def add_margin():

    import sys,os,glob,cv2,pdb
    import numpy as np
    root = '/Users/zhongz2/down/hidare_evaluation_from_Jinlin/HERE101_results_20250403'
    save_root = '/Users/zhongz2/down/hidare_evaluation_from_Jinlin/HERE101_results_20250404'

    for method in ['method3', 'method4', 'method5']:
        files = glob.glob(os.path.join(root, method, '*.jpg'))

        save_dir = os.path.join(save_root, method)
        os.makedirs(save_dir, exist_ok=True)

        for f in files:
            im = cv2.imread(f)
            H,W=im.shape[:2]
            n = int(W/1000)
            if int(W%1000)!=0:
                print(os.path.basename(f))
                pdb.set_trace()
            
            pad_img = 255*np.ones((H,10,3),dtype=np.uint8)
            res_imgs = []
            for i in range(n):
                res_imgs.append(im[:, i*1000:(i+1)*1000,:])
                res_imgs.append(pad_img)

            res_imgs = np.concatenate(res_imgs, axis=1)
            cv2.imwrite(os.path.join(save_dir, os.path.basename(f)), res_imgs)  # RGB --> BGR


def remove6cases():

    import sys,os,glob,cv2,pdb
    import numpy as np
    

    selected_prefixes = [
        '1034344-2',
        '1027436-1',
        '1034402-1',
        '564850-1',
        'liver-0',
        '1035020-2'
    ]

    root = '/Users/zhongz2/down/hidare_evaluation_from_Jinlin/HERE101_evaluation_20250409'
    
    for i in range(6):
        method = f'method{i+1}'
        if i==5:
            method = 'png'
        for prefix in selected_prefixes:
            files = []
            for postfix in ['.jpg', '.html', '.png']:
                files.extend(glob.glob(os.path.join(root, method, f'{prefix}{postfix}')))
            for f in files:
                os.system('rm -rf "{}"'.format(f))


def move6cases():

    import sys,os,glob,cv2,pdb
    import numpy as np
    

    selected_prefixes = [
        '1034344-2',
        '1027436-1',
        '1034402-1',
        '564850-1',
        'liver-0',
        '1035020-2'
    ]

    root = '/Users/zhongz2/down/hidare_evaluation_from_Jinlin/HERE101_results_20250407'
    save_root = '/Users/zhongz2/down/hidare_evaluation_from_Jinlin/HERE101_results_20250407_6cases'
    os.makedirs(save_root+'/twitter_data', exist_ok=True)

    for i in range(6):
        method = f'method{i+1}'
        if i==5:
            method = 'png'
        save_dir = os.path.join(save_root, method)
        os.makedirs(save_dir, exist_ok=True)
        for prefix in selected_prefixes:
            files = []
            for postfix in ['.jpg', '.html', '.png']:
                files.extend(glob.glob(os.path.join(root, method, f'{prefix}{postfix}')))
            for f in files:
                os.system('cp "{}" "{}"'.format(f, os.path.join(save_dir, os.path.basename(f))))

            twitter_dirs = []
            for f in files:
                if '.html' in f:
                    with open(f, 'r') as fp:
                        lines = fp.readlines()
                    for line in lines:
                        if 'Top-' in line:
                            s1 = line.find('target="_blank">')
                            s2 = line.find('</a></div>')
                            tid = line[s1+len('target="_blank">'):s2].split('/')[-1]
                            os.system('cp -r "{}" "{}"'.format(os.path.join(root, 'twitter_data', tid), os.path.join(save_root, 'twitter_data', tid)))

def generate_method2_html():

    import sys,os,glob,shutil
    import pandas as pd
    from natsort import natsorted


    selected_prefixes = [
        '1034344-2',
        '1027436-1',
        '1034402-1',
        '564850-1',
        'liver-0',
        '1035020-2'
    ]

    save_dir = '/Users/zhongz2/down/hidare_evaluation_from_Jinlin/HERE101_results_20250407/'

    files = natsorted(glob.glob('/Users/zhongz2/down/hidare_evaluation_from_Jinlin/HERE101_results_20250407/method2/*.html'))
    
    lines = []
    prefixes = []
    prefixes2 = []
    for f in files:
        fname = os.path.basename(f)
        prefix = os.path.splitext(fname)[0]
        lines.append('<a href="HERE101_eval/method2/{}.html" target="_blank">{}</a><br/>'.format(prefix, prefix))
        prefixes.append(prefix)

        if prefix not in selected_prefixes:
            prefixes2.append(prefix)

    # with open(os.path.join(save_dir, 'method2.html'), 'w') as fp:
    #     fp.writelines(lines)
    
    df = pd.DataFrame(prefixes, columns=['query'])
    df.to_excel(os.path.join(save_dir, 'scores.xlsx'), index=None)


    df2 = pd.DataFrame(prefixes2, columns=['query'])
    df2.to_excel(os.path.join(save_dir, 'scores_without_6cases.xlsx'), index=None)

def generate_r2_r4():

    import os,cv2,glob
    tif_dir = '/data/Jiang_Lab/Data/Zisha_Zhong/HERE101/tif'
    images_dir = '/data/Jiang_Lab/Data/Zisha_Zhong/HERE101/allpng_with_r2r4_20250403/png'
    os.makedirs(images_dir, exist_ok=True)

    for f in glob.glob(tif_dir+'/*.tif'):
        prefix = os.path.splitext(os.path.basename(f))[0]
        im = cv2.imread(f)
        H,W,C = im.shape
        if C!=3:
            print(prefix, ' not channels 3')
        if max(H,W)==4080:
            im = cv2.resize(im, dsize=None, fx=0.25,fy=0.25,interpolation=cv2.INTER_CUBIC)
        if max(H,W)==2040:
            im = cv2.resize(im, dsize=None, fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(images_dir, prefix+'.png' ), im)
        

    for f in glob.glob(images_dir+'/*.png'):
        prefix = os.path.splitext(os.path.basename(f))[0]
        im = cv2.imread(f)
        H,W=im.shape[:2]
        if max(H,W)!=1020:
            print(prefix, ' not 1020')
            break
        for scale in [2,4]:
            images_dir1 = '/data/Jiang_Lab/Data/Zisha_Zhong/HERE101/allpng_with_r2r4_20250403/png_r{}'.format(scale)
            os.makedirs(images_dir1, exist_ok=True)
            im1 = cv2.resize(im, dsize=None, fx=1/scale,fy=1/scale,interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(images_dir1, prefix+f'_r{scale}.png' ), im1)



# 20240804 HERE_CONCH
def main_here101_with_r2r4_CONCH(search_backbone):
    root = '/mnt/hidare-efs/data/HERE_assets'
    postfix=''  # for R0
    postfix='_r2r4' # for R2 and R4
    files = sorted(glob.glob(f'{root}/allpng_with_r2r4_20250403/png{postfix}/*.png'))
    print('files', files)
    # search_backbone = 'HERE_CONCH'
    save_dir = f'{root}/allpng_with_r2r4_20250403/search_results_{search_backbone}{postfix}'
    os.makedirs(save_dir, exist_ok=True)

    selected_prefixes = [
        '1034344-2',
        '1027436-1',
        '1034402-1',
        '564850-1',
        'liver-0',
        '1035020-2'
    ]

    for ind, f in enumerate(files):
        print('begin ', f)
        prefix = os.path.basename(f).replace('.png', '')
        # if prefix not in selected_prefixes:
        #     continue
        save_filename = os.path.join(save_dir, f'{prefix}.pkl')
        if os.path.exists(save_filename):
            continue
        params = {
            'k': 50,
            'search_project': 'KenData_20240814',
            'search_feature': 'before_attention',
            'search_method': 'faiss_IndexHNSWFlat_m32_IVFPQ_nlist128_m8',
            'socketid': '',
            'img_urls': [f],
            'filenames': [f],
            'resize': 0,
            'search_backbone': search_backbone
        }
        result = image_search_main(params=params)

        with open(save_filename, 'wb') as fp:
            pickle.dump(result, fp)

        # if ind==2:
        #     break



if __name__ == '__main__':
    # main_jiang_exp1()
    # main_here101_with_r2r4()

    # for search_backbone in ['HERE_PLIP', 'HERE_ProvGigaPath', 'HERE_CONCH']:
    #     main_here101_with_r2r4_CONCH(search_backbone)

    for search_backbone in ['HERE_CONCH']:
        main_here101_with_r2r4_CONCH(search_backbone)





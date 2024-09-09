from app_common import *

app = Flask(__name__,
            static_url_path='',
            static_folder='',
            template_folder='')


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
    pvalue_corr_thres = None
    try:
        gene_names = [vv for vv in params['gene_names'].split(',') if len(vv) > 0]
        cohensd_thres = float(params['cohensd_thres'])
        pvalue_corr_thres = float(params['pvalue_corr_thres'])
    except:
        pass

    status = ''
    if len(gene_names) == 0: 
        status = 'Invalid input gene names. \n'
    if cohensd_thres == None:
        cohensd_thres = 1.0
        status += 'Cohensd values should be float. Set it to 1.0. \n'
    if pvalue_corr_thres == None:
        pvalue_corr_thres = 0.1
        status += 'P-value(corrected) should be float. Set it to 0.1. \n'

    db_conn, db_cursor = get_db()
    hidare_table_str = f'cluster_result_table_20240814 as a, gene_table_20240814 as b, '\
        f'cluster_table_20240814 as c, cluster_setting_table_20240814 as d, st_table_20240814 as e'

    if len(gene_names) == 0:
        if cohensd_thres is None or pvalue_corr_thres is None:
            sql = f'select a.*, b.symbol, d.cluster_setting, e.prefix, c.cluster_label, c.cluster_info, f.note from {hidare_table_str} '\
                'where a.gene_id = b.id and c.id = a.c_id and c.cs_id = d.id and '\
                    'c.st_id = e.id order by a.cohensd desc limit 100;'
            try:
                db_cursor.execute(sql)
            except:
                db_conn, db_cursor = get_db()
                db_cursor.execute(sql)

        else:
            sql = f'select a.*, b.symbol, d.cluster_setting, e.prefix, c.cluster_label, c.cluster_info from {hidare_table_str} '\
                'where a.cohensd > %s and a.pvalue_corrected < %s and a.gene_id = b.id and c.id = a.c_id and c.cs_id = d.id '\
                    'and c.st_id = e.id order by a.cohensd desc limit 100;'
            try:
                db_cursor.execute(sql, (cohensd_thres, pvalue_corr_thres, ))
            except:
                db_conn, db_cursor = get_db()
                db_cursor.execute(sql, (cohensd_thres, pvalue_corr_thres, ))

    else:
        if cohensd_thres is None or pvalue_corr_thres is None:
            sql = f'select a.*, b.symbol, d.cluster_setting, e.prefix, c.cluster_label, c.cluster_info from {hidare_table_str} '\
                'where b.symbol in %s and a.gene_id = b.id and c.id = a.c_id and c.cs_id = d.id '\
                    'and c.st_id = e.id order by a.cohensd desc limit 100;'
            try:
                db_cursor.execute(sql, (gene_names,))
            except:
                db_conn, db_cursor = get_db()
                db_cursor.execute(sql, (gene_names,))
        else:
            sql = f'select a.*, b.symbol, d.cluster_setting, e.prefix, c.cluster_label, c.cluster_info from {hidare_table_str} '\
                'where b.symbol in %s and a.cohensd > %s and a.pvalue_corrected < %s and a.gene_id = b.id and c.id = a.c_id and '\
                    'c.cs_id = d.id and c.st_id = e.id order by a.cohensd desc limit 100;'
            try:
                db_cursor.execute(sql, (gene_names, cohensd_thres, pvalue_corr_thres))
            except:
                db_conn, db_cursor = get_db()
                db_cursor.execute(sql, (gene_names, cohensd_thres, pvalue_corr_thres))

    result = db_cursor.fetchall()
    if result is None or len(result) == 0:
        return {'current': 100, 'total': 100, 'status': 'Task completed!',
            'result': {'coxph_html_dict': '', 'response': '', 'images_shown_urls': ''}}

    result_df = pd.DataFrame(result, columns=['id', 'c_id', 'gene_id', 'cohensd', 'pvalue', 'pvalue_corrected',
                             'zscore', 'gene_symbol', 'cluster_setting', 'ST_prefix', 'cluster_label', 'cluster_info'])
    removed_columns = ['id', 'c_id', 'gene_id']
    result_df = result_df.drop(columns=removed_columns)

    sql = f'select note from image_table_20240814 where svs_prefix in %s'
    ST_prefixes = result_df['ST_prefix'].value_counts().index.values.tolist()
    try:
        db_cursor.execute(sql, (ST_prefixes, ))
    except:
        db_conn, db_cursor = get_db()
        db_cursor.execute(sql, (ST_prefixes, ))
    result = db_cursor.fetchall()
    ST_notes = {prefix: note[0] for prefix, note in zip(ST_prefixes, result)}
    db_conn.close()

    notes = []
    patch_annotations = []
    for rowid, row in result_df.iterrows():
        prefix = row['ST_prefix']
        cluster_setting = row['cluster_setting']
        cluster_label = int(float(row['cluster_label']))
        with open(os.path.join(ST_ROOT, cluster_setting, prefix, prefix+'_cluster_data.pkl'), 'rb') as fp:
            cluster_labels = pickle.load(fp)['cluster_labels']
        with open(os.path.join(ST_ROOT, cluster_setting, prefix, prefix+'_patches_annotations.json'), 'r') as fp:
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


@app.route('/image_search', methods=['POST', 'GET'])
def image_search():
    params = request.get_json()
    return image_search_main(params)


app.wsgi_app = ProxyFix(app.wsgi_app)
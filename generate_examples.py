"""Reproducible, two-dimensional k-means teaching examples and local HTML gallery."""
from pathlib import Path
from html import escape
import base64
import csv
import json
import sys
import struct

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.datasets import make_moons, make_circles
from sklearn.metrics import adjusted_rand_score, rand_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'style'/'scripts'))
from plot_style import plot_context, save_panel

COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9']
NOISE_COLOR = '#777777'
SEED = 20260930
UNCERTAINTY = 'Raw synthetic observations; no inferential confidence intervals apply.'


def datasets():
    rows = []
    def add(key, title, X, y, k, lesson, source=None, **extra):
        rows.append(dict(key=key, title=title, X=np.asarray(X, float), truth=np.asarray(y, int),
                         k=k, lesson=lesson, source=source, **extra))
    def blobs(seed, centers, scales, counts):
        rng = np.random.default_rng(seed)
        X = np.vstack([rng.normal(c, scale, (n, 2)) for c, scale, n in zip(centers, scales, counts)])
        return X, np.repeat(np.arange(len(centers)), counts)

    X, y = blobs(SEED+1, [[-2.5,0],[2.5,0]], [.42,.42], [80,80])
    add('separated_round', 'Separated round groups', X, y, 2,
        'A favorable case: similarly sized, compact groups are well separated. A nearby mean is a useful summary of each group.', 'PDF pages 1–3 · slide 11')

    X, y = blobs(SEED+2, [[-2,0],[2,0]], [.20,1.30], [90,90])
    add('unequal_spread', 'Unequal spread', X, y, 2,
        'Both groups contain 90 observations, but one is much more diffuse. The nearest-center boundary gives some diffuse-group points to the compact group. Equal spread is helpful, not a formal requirement.', 'PDF pages 4–5 · slide 12')

    X, y = blobs(SEED+3, [[-2.8,0],[2.8,0]], [1.15,1.15], [10,300])
    add('unequal_counts', 'Unequal group sizes', X, y, 2,
        'The generating groups have the same population spread but 10 versus 300 observations. K-means can reduce total squared error by splitting the large group and grouping the small population with part of it.', 'PDF pages 6–7 · slide 13')

    X, y = blobs(SEED+4, [[-1.25,0],[1.25,0]], [[.12,3.1],[.12,3.1]], [90,90])
    add('elongated_groups', 'Long, narrow groups', X, y, 2,
        'The generating groups are the left and right columns. Their large vertical spread makes a top-versus-bottom partition cheaper in squared distance. K-means does not identify groups from their shape alone.', 'PDF pages 8–9 · slide 14')

    X, y = blobs(SEED+5, [[-.55,0],[.55,0]], [.72,.72], [90,90])
    add('overlapping_groups', 'Overlapping groups', X, y, 2,
        'The generating populations overlap. The algorithm creates a hard nearest-center boundary, but nearby observations can have different hidden origins. Even a sensible geometric split cannot perfectly recover those labels.', 'PDF pages 10–12 · slide 15')

    X, y = make_moons(n_samples=180, noise=.045, random_state=SEED+6)
    add('two_moons', 'Two interlocking moons', X, y, 2,
        'Two curved groups are easy to follow by eye. With two centers, k-means uses a straight dividing boundary and cuts across the moons.')

    X, y = make_circles(n_samples=180, factor=.43, noise=.027, random_state=SEED+7)
    add('nested_rings', 'Nested rings', X, y, 2,
        'The hidden groups are the inner and outer rings. K-means divides space into nearest-center regions, so it cuts across both rings instead of separating inside from outside.')

    X, y = blobs(44, [[-3,-1.5],[0,3],[3,-1.5]], [.35,.35,.35], [60,60,60])
    for k, key, title, lesson in [
        (2,'too_few_clusters','Choosing k: too few', 'These data were generated from three compact groups. Asking for two clusters forces k-means to merge two of them.'),
        (3,'matching_k','Choosing k: a good match', 'The same points with k=3: one center per generating group. This is the reference for the k=2, k=4, and poor-initialization examples.'),
        (4,'too_many_clusters','Choosing k: too many', 'The same points with k=4: one generating group is split. Extra centers reduce squared error, so a lower objective alone does not establish the right number of groups.')]:
        add(key,title,X.copy(),y.copy(),k,lesson)
    add('poor_initialization', 'A poor initialization', X.copy(), y.copy(), 3,
        'The same three-group dataset, but a deliberately poor starting arrangement and only one run. Two centers stay in one group while one center serves the other two. Compare the objective with the multi-start k=3 result above; multiple starts help, but do not guarantee a global optimum.',
        init=np.array([[-3,-1.8],[-3,-1.2],[1.5,.75]]), n_init=1)

    rng = np.random.default_rng(SEED+8)
    nuisance = np.linspace(-6,6,70)
    jitter = rng.normal(0,.12,70)
    X = np.vstack([np.column_stack((-1.35+jitter,nuisance)), np.column_stack((1.35+jitter,nuisance))])
    y = np.repeat([0,1],70)
    add('feature_scale_raw', 'Feature scaling: original units', X, y, 2,
        'The groups differ in feature 1, but feature 2 has a much larger numerical range. Raw Euclidean distances are dominated by feature 2, so k-means splits along that variation.')
    scaler = StandardScaler().fit(X)
    add('feature_scale_standardized', 'Feature scaling: standardized', scaler.transform(X), y, 2,
        'The same observations after standardizing each feature to unit variance. All three panels show standardized coordinates, exactly as supplied to k-means. Scaling recovers the generating groups here; it changes the question and is not universally the right choice.',
        raw_X=X, scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist(), standardized=True)

    X, y = blobs(SEED+9, [[-2,0],[2,0]], [.40,.40], [60,60])
    X = np.vstack((X,[[0,25]])); y = np.r_[y,-1]
    add('distant_outlier','One distant outlier',X,y,2,
        'The grey point is a designated outlier, not a third generating group. K-means has no reject option: it uses one of the two centers for that point and merges the two main groups. The reported agreement scores exclude the outlier; the fit includes it.')

    rng = np.random.default_rng(SEED+10)
    X = rng.normal(size=(160,2))
    add('single_population','Clusters without separate populations',X,np.zeros(len(X),int),3,
        'All observations come from one Gaussian population. K-means still returns three groups when asked. Colored partitions alone are not evidence for distinct underlying populations.')
    return rows


def fit(row):
    model = KMeans(n_clusters=row['k'], init=row.get('init','k-means++'),
                   n_init=row.get('n_init',50), random_state=42,
                   algorithm='lloyd', max_iter=1000, tol=1e-8).fit(row['X'])
    y, labels = row['truth'], model.labels_
    valid = y >= 0
    groups = np.unique(y[valid])
    contingency = np.array([[np.sum((labels==p)&(y==g)) for g in groups] for p in range(row['k'])])
    pred_ids, truth_ids = linear_sum_assignment(-contingency)
    mapping = {int(p):int(groups[g]) for p,g in zip(pred_ids,truth_ids)}
    free = iter(c for c in range(len(COLORS)) if c not in mapping.values())
    for p in range(row['k']):
        if p not in mapping: mapping[p]=next(free)
    # Matching changes display colors only; truth never enters the k-means fit.
    matched = np.array([mapping[int(p)] for p in labels])
    row.update(model=model, labels=labels, matched=matched, mapping=mapping,
               ari=float(adjusted_rand_score(y[valid],labels[valid])),
               ri=float(rand_score(y[valid],labels[valid])))
    return row


def limits(X):
    span=float(np.ptp(X,axis=0).max())
    mids=(X.max(axis=0)+X.min(axis=0))/2
    return [(float(m-.60*span),float(m+.60*span)) for m in mids]


def figure(row, stage):
    X, y = row['X'], row['truth']
    f, ax = plt.subplots(figsize=(4.3,3.5))
    labels = y if stage=='truth' else row['matched']
    if stage=='input': colors=['#333333']*len(X)
    else: colors=[NOISE_COLOR if v<0 else COLORS[v] for v in labels]
    ax.scatter(X[:,0], X[:,1], c=colors, s=20, alpha=.86, linewidths=0, zorder=2)
    handles=[]
    if stage!='input':
        for v in sorted(set(labels)):
            label = 'Designated outlier' if v<0 else ('Group ' if stage=='truth' else 'Cluster ')+chr(65+v)
            handles.append(Line2D([],[],marker='o',ls='',color=NOISE_COLOR if v<0 else COLORS[v],label=label))
    if stage=='kmeans':
        centers=row['model'].cluster_centers_
        ax.scatter(centers[:,0],centers[:,1],c='black',marker='X',s=80,
                   linewidths=.7,edgecolors='white',zorder=4)
        handles.append(Line2D([],[],marker='X',ls='',color='black',label='Fitted center'))
    xlim,ylim=limits(X)
    ax.set_xlim(xlim);ax.set_ylim(ylim)
    ax.set_aspect('equal',adjustable='box')
    ax.xaxis.set_major_locator(MaxNLocator(3));ax.yaxis.set_major_locator(MaxNLocator(3))
    unit = ' (z-score)' if row.get('standardized') else ''
    ax.set_xlabel('Feature 1'+unit);ax.set_ylabel('Feature 2'+unit)
    title={'truth':'Ground truth','input':'Unlabelled input','kmeans':f"k-means (k={row['k']})"}[stage]
    ax.set_title(title,fontsize=17,pad=10)
    counts = ', '.join(f"{'outlier' if g<0 else 'group '+chr(65+g)} n={int(np.sum(y==g))}" for g in sorted(set(y)))
    units='standardized z-scores' if row.get('standardized') else 'arbitrary feature units'
    meaning={'truth':'Colors show the generating labels, which are not supplied to k-means.',
             'input':'All points are neutral: these are the exact coordinates supplied to k-means, with labels withheld.',
             'kmeans':'Colors show the fitted assignments; black X marks show fitted centers. Colors are matched to ground truth only after fitting for comparison; cluster names are arbitrary.'}[stage]
    cap=(f"{row['title']}. {meaning} n={len(X)} synthetic observations ({counts}); {units}. "
         'All three views use identical coordinates, axis limits, physical unit scales, point sizes, and ordering. '
         f"Euclidean k-means, k={row['k']}, {row.get('n_init',50)} initialization(s), random_state=42. "
         f"{row['lesson']} Reference: {row['source'] or 'Additional teaching example'}. These are new synthetic illustrations, not digitized slide data.")
    save_panel(f,ROOT/'figures'/f"{row['key']}_{stage}.png", caption=cap,uncertainty=UNCERTAINTY,
               legend_entries=(handles,[h.get_label() for h in handles]))
    plt.close(f)


def export(rows):
    (ROOT/'data').mkdir(exist_ok=True);(ROOT/'figures').mkdir(exist_ok=True)
    summary=[]
    for row in rows:
        for stage in ['truth','input','kmeans']:figure(row,stage)
        names=['feature_1','feature_2','ground_truth','kmeans_label','display_label']
        arr=np.column_stack((row['X'],row['truth'],row['labels'],row['matched']))
        if 'raw_X' in row:
            names+=['original_feature_1','original_feature_2'];arr=np.column_stack((arr,row['raw_X']))
        np.savetxt(ROOT/'data'/f"{row['key']}.csv",arr,delimiter=',',header=','.join(names),comments='',fmt='%.12g')
        m=row['model']
        entry={k:row[k] for k in ['key','title','k','lesson','source','ari','ri']}
        entry.update(n=len(row['X']),counts={str(int(g)):int(np.sum(row['truth']==g)) for g in np.unique(row['truth'])},
                     inertia=float(m.inertia_),iterations=int(m.n_iter_),centers=m.cluster_centers_.tolist(),
                     n_init=row.get('n_init',50),initialization=row['init'].tolist() if 'init' in row else 'k-means++',
                     color_mapping=row['mapping'],scored_points=int(np.sum(row['truth']>=0)),axis_limits=limits(row['X']))
        if 'raw_X' in row:entry.update(scaler_mean=row['scaler_mean'],scaler_scale=row['scaler_scale'])
        summary.append(entry)
        print(f"{row['key']}: k={row['k']}, ARI={row['ari']:.3f}, inertia={m.inertia_:.2f}",flush=True)
    (ROOT/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (ROOT/'metrics.csv').open('w') as f:
        fields=['key','n','k','ari','ri','inertia','n_init','scored_points']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(summary)
    gallery(summary)


def gallery(rows):
    assets={};cards=[]
    def panel(row,stage,label):
        name=f"{row['key']}_{stage}.png"
        raw=(ROOT/'figures'/name).read_bytes()
        w,h=struct.unpack('>II',raw[16:24])
        assets[name]=dict(mime='image/png',base64=base64.b64encode(raw).decode('ascii'),width=w,height=h)
        status='status-'+name.replace('.','-')
        return (f'<figure><img src="figures/{name}" alt="{escape(row["title"])}: {escape(label)}">'
                f'<figcaption>{label}</figcaption><div class="actions">'
                f'<button data-copy-image="{name}" aria-label="Copy {escape(row["title"])}: {escape(label)}" aria-describedby="{status}">Copy hi-res PNG</button>'
                f'<a href="figures/{name}" download>Download PNG</a><a href="figures/{name}" target="_blank" rel="noopener">Open PNG</a></div>'
                f'<p class="copy-status" id="{status}" role="status" aria-live="polite"></p></figure>')
    for i,row in enumerate(rows,1):
        provenance=row['source'] or 'Additional teaching example'
        score_scope=' · scores on inliers only' if row['scored_points']!=row['n'] else ''
        cards.append(f'<section><div class="row-heading"><h2>{i}. {escape(row["title"])}</h2><span class="badge">{escape(provenance)}</span></div>'
            +f'<p>{escape(row["lesson"])}</p><div class="panels">'
            +panel(row,'truth','1. Generating groups')+panel(row,'input','2. What the algorithm sees')+panel(row,'kmeans','3. Fitted assignments · black × = centers')
            +f'</div><p class="metrics">n = {row["n"]} · k = {row["k"]} · ARI = {row["ari"]:.3f} · Rand index = {row["ri"]:.3f}{score_scope} · <a href="data/{row["key"]}.csv">Data CSV</a></p>'
            +f'<details><summary>Fit details</summary><p>Sum of squared distances = {row["inertia"]:.3f}; {row["n_init"]} initialization(s); {row["iterations"]} iterations for the selected fit. Compare this objective only for the same observations and feature scaling.</p></details></section>')
    template=(ROOT/'gallery_template.html').read_text()
    template=template.replace('__ROWS__','\n'.join(cards)).replace('__ASSETS__',json.dumps(assets,separators=(',',':'))).replace('__SCRIPT__',(ROOT/'gallery_ui.js').read_text())
    (ROOT/'gallery.html').write_text(template)


if __name__=='__main__':
    rows=[fit(r) for r in datasets()]
    with plot_context():export(rows)

# K-means teaching demos

Open **gallery.html** in a browser. Every example occupies one row with three separate panels:

1. **Ground truth:** color denotes the synthetic generating group.
2. **Unlabelled input:** the exact same observations and coordinates, without group labels.
3. **K-means result:** fitted assignments, with black X marks for fitted centers.

There are 15 examples and 45 high-resolution PNG panels. Each PNG is 1290 × 1050 pixels at 300 DPI. The gallery includes copy, download, and open buttons for each panel; copying depends on browser clipboard support and permissions. PNG bytes are embedded in the HTML so that copying does not require fetching a local file. All panels remain visible side by side; narrow screens scroll horizontally.

## Examples

The first five rows qualitatively reproduce all distinct examples in the supplied `ns5007-dim-reduction-short.pdf` (12 pages containing slide builds numbered 11–15):

- Separated round groups.
- Unequal spread, with equal observation counts.
- Unequal observation counts, with equal generating spread.
- Long, narrow groups.
- Overlapping groups.

Ten added rows show interlocking moons, nested rings, k=2/3/4 on the same three-group dataset, poor initialization on that dataset, original versus standardized feature scales, a distant outlier, and a single Gaussian population partitioned into three clusters.

See **SOURCE_NOTES.md** for the page mapping and interpretation notes. These are newly generated synthetic points, not digitizations or copies of the slide figures.

## Scientific conventions

- Standard rows use Euclidean k-means, k-means++ initialization, 50 starts, random_state=42, and the best objective among those starts. This does not guarantee the global optimum.
- The poor-initialization row deliberately uses specified starting centers and one start. Its data are identical to the three k-choice rows.
- The class labels never enter fitting. Hungarian matching changes only display colors after fitting; cluster numbers and colors have no intrinsic meaning.
- Every row shares point coordinates, axis limits, equal physical scale per unit, point sizes, and point order across its three panels. There is no display jitter and no axis distortion.
- The standardized example displays standardized coordinates in all three panels, matching the actual algorithm input. Its CSV also includes the original coordinates. All other examples use unscaled coordinates.
- The outlier is truth label -1 and has no assumed generating cluster. It is included in fitting and excluded from the reported Rand index and adjusted Rand index (ARI), which are computed on the 120 inliers only.
- All other agreement scores include every observation. The single-population example has one true generating population even though k-means is asked for k=3.
- The objective is the sum of squared distances to fitted centers. Compare it only across fits to the same observations on the same feature scales. Increasing k can reduce this objective without recovering the intended groups.
- These are raw synthetic observations, so confidence intervals are not applicable.

## Reproduce

```sh
python -m pip install -r requirements-lock.txt
python generate_examples.py
```

The requirements file records the package versions used to produce these files. The script uses fixed dataset seeds based on 20260930; the three-group comparison uses seed 44. The plotting helper and style are included, so no personal skill installation is needed.

## Files

- `generate_examples.py`: datasets, actual k-means fits, figures, CSVs, and gallery generation.
- `gallery.html`, `gallery_template.html`, `gallery_ui.js`: generated gallery, template, and clipboard behavior.
- `figures/*_truth.png`, `*_input.png`, `*_kmeans.png`: separate panels for slide assembly.
- `figures/*.caption.md`: standalone captions with sample counts and methods.
- `figures/*-legend.png`: detached legends for ground-truth groups and predicted clusters.
- `data/*.csv`: algorithm inputs, true labels, fitted labels, and display labels.
- `results.json`: full fit metadata, means, initializations, axis limits, and scores.
- `metrics.csv`: compact table of fit statistics.
- `style/`: portable plotting-style snapshot.

The original PDF is not bundled; the source notes identify the slide examples reproduced qualitatively.

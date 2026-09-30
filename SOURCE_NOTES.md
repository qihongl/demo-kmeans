# Source mapping and teaching notes

Source: user-provided **ns5007-dim-reduction-short.pdf**, 12 PDF pages. The visible slide numbers are 11–15; multiple PDF pages are successive builds of the same slide. All 12 pages were visually inspected. No readable text layer was present.

## Every distinct slide example

| PDF pages | Visible slide | Gallery row | Reproduced behavior |
| --- | --- | --- | --- |
| 1–3 | 11 | Separated round groups | Two separated, similarly sized compact groups are recovered. |
| 4–5 | 12 | Unequal spread | Equal counts but different spread; part of the diffuse group is assigned to the compact group's cluster. |
| 6–7 | 13 | Unequal group sizes | Many more observations in one group; both fitted centers are used to split the dominant group. |
| 8–9 | 14 | Long, narrow groups | Left/right elongated generating groups are partitioned roughly top/bottom. |
| 10–12 | 15 | Overlapping groups | A hard geometric partition does not recover all overlapping generating labels. |

The source pages motivate qualitative geometry and behavior. None of the source points, coordinates, exact sample sizes, or fitted centers were recoverable from a numerical table; all points here are synthetic, and all displayed assignments are newly computed with scikit-learn KMeans.

## Added examples

- **Two moons and nested rings:** convex nearest-center regions cannot follow these nonconvex group shapes. For k=2 the nearest-center boundary is a straight line.
- **k=2, 3, 4:** identical data, distinct requested numbers of clusters; useful for discussing merging, splitting, and why training squared error alone cannot select k.
- **Poor initialization:** identical data and k=3, deliberately poor initial centers, n_init=1. This separates an optimization failure from the geometric limitations seen in the other rows. Initial centers and final objective are saved in results.json.
- **Feature scaling:** identical original observations are shown before and after StandardScaler. Coordinates in the second row are the standardized input, not a picture drawn in a different space from the fitted model. Scaling helps this constructed example; it does not guarantee recovery in general.
- **Outlier:** one remote observation consumes a center because squared distance gives it high weight. There is no outlier rejection in ordinary k-means.
- **Single population:** specifying k creates a partition even without separate generating populations. A colored plot alone does not establish distinct categories.

## Interpretation cautions relevant to teaching

The textbook description of favorable k-means data as round, similarly sized, and well separated is a useful guide, not a set of necessary conditions. The relevant mathematical objective is total within-cluster squared distance. Several failures shown here have lower squared error than the generating-label partition.

Agreement with generating labels measures recovery of this synthetic construction. In real unsupervised problems those labels may be unavailable, ambiguous, or unrelated to the geometric partition of interest. All color matching is performed after fitting and leaves the partition and its scores unchanged.

The displayed solutions are the results of a reproducible numerical run. Standard runs select the lowest objective among 50 starts; they are not certified global optima.

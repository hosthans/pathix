# Graph corpora: what each pipeline sets

The six `*.yaml` files in this folder are **flattened snapshots** of the runtime configs in
`configs/graph_runtime/` (no `extends`, no `build_config`: every setting is written out). Each one was
verified to load to exactly the same values as its original with `build_graph._load_config`
(runtime, `BuildConfig`, `CreationConfig`, explicitly-set fields, effective model overrides, output
folder name). Build the corpora with these files (`build_graph.py --config <file>`); they write to the same folders as the originals.

| Setting | G-aug | G-noaug | G-R50 | C-aug | C-noaug | C-R50 |
|---|---|---|---|---|---|---|
| Dataset | GSDB | GSDB | GSDB | CAMELYON17 | CAMELYON17 | CAMELYON17 |
| Encoder | Prov-GigaPath | Prov-GigaPath | ResNet-50 | Prov-GigaPath | Prov-GigaPath | ResNet-50 |
| Macenko normalisation (reference: GSDB scanner-1 train / CAMELYON train) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Stain augmentation (σ 0.2, p 0.5) | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ |
| Gaussian blur (σ ≤ 1.2, p 0.2) | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ |
| JPEG compression (quality 40–95, p 0.2) | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ |
| Fixed seed for patch perturbation (`transform_seed: 42`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Bit-exact builds (`feature_redundancy_check`, `blas_num_threads: 1`, `cv2_num_threads: 1`) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Patch 256 px, stride 256, level 0, tissue fraction ≥ 0.08 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Physically calibrated patch size (`auto_detect_target_patch_size_um`) | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| Geometric regions (≤ 100 patches) | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| Annotation-driven regions | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ |
| Patch–patch edges: `semantic_equality` | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| Patch–patch edges: kNN (k = 12) | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ |
| Region–region edges: kNN (k = 5) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Tissue mask computed on the fly | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| Precomputed tissue masks | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ |
| Annotation format | GeoJSON | GeoJSON | GeoJSON | ASAP | ASAP | ASAP |
| Annotation label source | `inflammation_status` | `inflammation_status` | `inflammation_status` | `group` | `group` | `group` |
| Slides without annotation are built | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ |
| Label fields on the graph | status, tissue_type | status, tissue_type | status, tissue_type | status, subtype, stage | status, subtype, stage | status, subtype, stage |

G = GSDB, C = CAMELYON17; aug = with the three image-level augmentations, noaug = Macenko only, R50 = ResNet-50 encoder
(augmented). Corpora of one dataset differ **only** in the encoder (R50) or in the augmentations (noaug); GSDB and
CAMELYON17 differ in the eight rows from "Physically calibrated patch size" down to "Label fields".

# Pathix

**Dynamic and modular graph construction from Whole Slide Images (WSIs) for computational pathology.**

Pathix is a research framework developed as part of a Master's thesis to provide a flexible foundation for constructing graph representations from whole-slide images. Its modular architecture enables different graph creation pipelines, allowing researchers to combine, extend, and adapt individual processing stages to their own needs and research questions.

Rather than enforcing a fixed graph representation, Pathix makes graph construction configurable — from tissue and region extraction to feature generation and graph assembly.

## Key Features

- **Modular pipelines** — Combine and customize different graph creation strategies.
- **Dynamic graph construction** — Configure how tissue regions, nodes, edges, and features are generated.
- **Extensible architecture** — Integrate custom methods, feature extractors, and processing stages.
- **Reproducible experiments** — Configurable workflows, seeded training, and deterministic graph generation.
- **Flexible downstream learning** — Support for patch-based CNNs, MIL, and graph neural networks.

## Research Application

Pathix is developed and evaluated in the context of domain generalization in computational pathology:

| Dataset | Task | Domain shift |
|---|---|---|
| GSDB | Gastric inflammation | Scanner 1 → Scanner 2 |
| CAMELYON17 | Lymph-node metastasis | Center shift |

The thesis experiments compare patch-based CNNs, multiple-instance learning (MIL), and graph-based models, including Tissue-Region-Net (TRN, `tissue_region_net`).

## Project Structure

```text
pathix/
├── configs/                 Experiment and graph configurations
├── data/                    Data, tiles, graphs, and splits (gitignored)
├── docs/                    Pipeline and experiment documentation
├── notebooks/analysis/      Experimental analysis
├── scripts/                 Helper and analysis scripts
└── src/
    ├── entrypoint/          CLI commands
    ├── graph_creation_new/  Graph construction methods
    ├── pipelines/           Processing pipelines
    ├── tissue_region_extraction/
    ├── models/              Learning architectures
    ├── dataset/             Dataset handling
    ├── core/                Core components
    └── utils/
```

## Documentation

| Document | Description |
|---|---|
| [Pipeline](docs/pipeline.md) | Setup, data preparation, graph construction, models, and execution |
| [Experiments](docs/experiments/cross_domain_experiments.md) | E1–E4: commands, configurations, training, evaluation, and results |
| [Experiment Details](docs/experiments/cross_domain_experiments_details.md) | Protocols, methodological differences, and validity considerations |
| [Artifacts](docs/artifacts/README.md) | The thesis artifacts (framework, TRN, DG framework) and supporting artifacts: purpose, code locations, interfaces, configuration keys |

## Reproducibility

Training supports explicit random seeds and deterministic kernels via `--deterministic`. Graph construction is bit-exact for identical inputs and configurations, verified through `feature_redundancy_check`. FP16 training runs are not guaranteed to be bit-identical across retraining.

---

*Pathix — A flexible foundation for graph-based computational pathology.*

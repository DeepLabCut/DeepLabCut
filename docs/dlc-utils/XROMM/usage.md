(file:xamalab-dlc-integration)=

# XROMM + DeepLabCut local integration

These notes describe how this repository is used in the local 3-repo XROMM workflow together with `../XROMM_DLCTools` and `../xmalab`.

> Contributed by [@homfunc](https://github.com/homfunc)

## 1) Expected local layout

Recommended sibling checkout layout:

- `XROMM_DLCTools/`
- `DeepLabCut/`
- `xmalab/`
  `XROMM_DLCTools/pyproject.toml` maps its optional `dlc` dependency group to this repository through `tool.uv.sources`.

## 2) DeepLabCut’s role in the workflow

Within the integrated workflow, DeepLabCut provides:

- project creation / dataset generation support
- video analysis / prediction entrypoints
- the local import target used by `XROMM_DLCTools`
- synthetic smoke coverage through the baseline harness
  The current local integration suite also uses this repo to verify that the newer workflow service in `XROMM_DLCTools` still interoperates with a sibling DeepLabCut checkout.

## 3) Local setup for this repo

Standard developer setup:

```bash
uv sync --group dev
```

When working from `../XROMM_DLCTools`, enable the sibling import path there with:

```bash
uv sync --group dlc
```

## 4) Integration validation from XROMM_DLCTools

Run these commands from `../XROMM_DLCTools`:

```bash
uv run python scripts/baseline_harness.py --scenario deeplabcut_repo_smoke --output-dir baseline_artifacts/deeplabcut_smoke --deeplabcut-repo ../DeepLabCut
```

Full multi-repo suite:

```bash
uv run python scripts/baseline_harness.py --scenario all --output-dir baseline_artifacts/integration_all --deeplabcut-repo ../DeepLabCut --xmalab-repo ../xmalab
```

True end-to-end local workflow scenario:

```bash
uv run python scripts/baseline_harness.py --scenario phase3_local_workflow_e2e --output-dir baseline_artifacts/e2e_local_workflow --deeplabcut-repo ../DeepLabCut --xmalab-repo ../xmalab
```

## 5) Compatibility notes

The local workflow integration path expects this repo to remain importable in “lite mode” when GUI dependencies are unavailable, and relies on the public `deeplabcut` import surface plus the synthetic project helpers under `examples/utils.py`.

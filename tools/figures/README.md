# tools/figures

Declarative figure registry plus one build command. Figures are built by `tools/figures/figs/*.py`
modules, registered one YAML file each in `tools/figures/registry/`.

## The one command

```bash
cd /home/ayp/project/trm && CUDA_VISIBLE_DEVICES="" uv run --frozen --no-sync python -m tools.figures.build_all \
    --slug <slug> --cache-dir <dir> (--lab-copy-root <dir> | --no-lab-copy) \
    [--registry-dir <dir>] [--out-root <dir>] [--only <id> ...] [--list] \
    [--refresh-cache --families <yaml> [--jobs N]]
```

- Defaults: registry-dir `tools/figures/registry`, out-root `reports/figures`; output is `<out-root>/<slug>`.
- Exactly one of `--lab-copy-root` / `--no-lab-copy`, else exit 2 before anything is built or written.
- `--refresh-cache` first runs `python -m tools.figures.extract_curves --families <yaml> --out <cache-dir> --jobs N`
  (WANDB_MODE=disabled, all WANDB_* dirs in one fresh temp directory); failure stops with exit 1.
- Each figure runs in its own subprocess (`run_one`) with `CUDA_VISIBLE_DEVICES=""` and `MPLBACKEND=Agg`.
  A failing figure does not stop the others.
- Exit codes: 0 all good; 1 a figure, the extractor or the lab copy failed; 2 usage error; 3 output (or lab copy) directory exists and is not empty.
- If any figure fails, the lab copy is skipped (a broken build never reaches the lab tree).

## Never write into an existing figure directory

A build refuses (exit 3) an output directory that is not empty, and the lab copy destination `<lab-copy-root>/<slug>`
must not exist. Nothing in this package deletes or overwrites; pick a new slug for every build.

## Registry schema (one YAML per figure, file stem = id)

| key | meaning |
|---|---|
| `id` | e.g. `RF-03-curves-budget`; equals the file stem |
| `module` | dotted path of a module exposing `build(ctx)` |
| `title` | plain title |
| `langs` | subset of `[en, ko]` |
| `print_width_in` | printed width in inches (float); `ctx.save` fails if any text would print below 7 pt |
| `inputs` | list of `{role: cache, family, file}` or `{role: file, path: <absolute>}` |
| `protocol` | `{}` or `{metric_test, metric_train}` |
| `cohorts`, `report_refs` | lists of strings |

Unknown or missing keys are rejected. `python -m tools.figures.registry --dir <dir> --list` lists the entries.

## Output files in `<out-root>/<slug>/`

- `<id>.<lang>.png` (300 dpi) and `<id>.<lang>.pdf` (no creation date), per language.
- `<id>.values.csv` (language-independent), columns `figure_id,panel,series,run_name,wandb_id,project,cohort,metric,step,value,source_ref`, at least one row.
- `<id>.source.json` (figure_id, module, module_sha256, registry_sha256, inputs with sha256 and rows_used, protocol, cohorts, run_names, wandb_ids, n_values_rows).
- `<id>.<name>.csv` extras; `build_manifest.json` (return code, seconds, stderr tail, sha256 of every output); `FIGURE_INDEX.json`
  (`slug`, `cache_manifest_sha256`, `figures.<id>` with files, run_names, wandb_ids, projects, cohorts, protocol, values_csv, source_json; failed figures are omitted).
  `cache_manifest_sha256` is `null` when the cache has no `MANIFEST.files.csv`.
- With a lab copy: `LAB_COPY.sha256` in both places.

## Tracing a number back

A number on a figure is a row of `<id>.values.csv`: `run_name` + `wandb_id` + `metric` + `step` (+ `project`, `cohort`).
`<id>.source.json` gives the input file hashes and the module and registry hashes; for figures not drawn from runs,
`source_ref` names the timeline or claim row. Cache CSVs are read through `io.load_cache_csv`, which checks the sha256 against `MANIFEST.files.csv`.

## Run modules as `python -m`

Always `python -m tools.figures.<module>` from `/home/ayp/project/trm`, never a script file from inside this directory:
there `io.py` would shadow the standard library's `io`, and `registry.py` sits beside the data directory `registry/`.

`style.py` is the only place that defines rcParams and the palette (vendored from the NAS `viz_style.py`; header states path and sha256).

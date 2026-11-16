# Cross-Room Few-Shot Adaptation — CS60055 Hackathon Challenge 2

Student-owned Python pipeline: raw Nexmon PCAP / amplitude CSV → common 256-column form → quality control →
`(150, 256)` windows → self-supervised encoder → room fingerprint → few-label cross-room adaptation
(A→B and B→A, K ∈ {1, 5, 10, 20}) with environment-aware routing → validation results, required figures and
an episode runner.

```
 Exp-2 .pcap ──► UDP payloads ► Nexmon header check ► packed-float CSI decode ► |CSI| ┐
                                                                                       ├► QC ► intermediate_csv/ (256 cols/capture)
 Exp-3 .csv  ──► header/width check —───────────────────────────────────────────────────┘        │
                                                                                                ▼
                 windows (150×256, window_XXXXX.npy + manifests/windows.csv provenance) ◄── capture-level dev/val split
                                   │ development-fitted normaliser
                  ┌────────────────┼──────────────────────────┐
                  ▼                ▼                          ▼
        Task 1 room fingerprint   Task 2 SSL encoder      Task 3 per-room alignment + K-shot head
        (Room A vs B)             (no labels) + probes    + predicted-room routing  → predictions
```

## 1. Setup

```bash
# environment (pick one)
conda env create -f environment.yml && conda activate csi-crossroom
#   or
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

## 2. Data (kept outside git, read-only)

```bash
mkdir -p data/raw
# unzip WITHOUT macOS junk, the source authors' MATLAB folder, the Experiment-3 notebook and result images
unzip -q /path/to/1.zip -d data/raw -x "*__MACOSX*" "*.DS_Store" "*matlab code*" "*.ipynb" "*/Results/*" "*.png"
```

`paths.raw_root` in `configs/experiment.yaml` points at `data/raw`; Experiment-2's `csidata/` and
Experiment-3's `Data/` are discovered at any depth (the supplied zip nests them as `1/Experiment-2/Experiment-2/...`).
Nothing is ever written to the raw data; all derived files go to `workspace/`.
Check the layout first (optional): `python scripts/inspect_data.py`.

## 3. Run

```bash
python run_pipeline.py --stage all          # ingest → windows → ssl → task1 → task2 → task3 (+ figures, predictions, export)
```
Stages can be run one by one: `--stage ingest|windows|ssl|task1|task2|task3|export`; already-finished stages
are skipped unless `--force`. Config values can be overridden, e.g. a quick smoke test:
`python run_pipeline.py --stage all --set ssl.epochs=3 "task3.seeds=[0,1]"`.

Run an episode (folder or single file of `.pcap` / `.csv` / `.npy`; the room is *not* given, it is predicted):

```bash
python run_episode.py --episode /path/to/episode --out predictions.csv
```
→ `predictions.csv` (`window_id, room, label, confidence`) and `predictions_provenance.csv`.
It needs the trained artifacts (`workspace/artifacts/`, a few MB — keep them in the repo or regenerate with the pipeline).

## 4. Outputs (`workspace/`)

| Path | Content |
|---|---|
| `intermediate_csv/room_{A,B}/{dev,val}/<activity>/<capture>.csv` | common intermediate form: one row per valid observation, columns `subcarrier_001…256` |
| `manifests/captures.csv`, `qc_summary.json` | per-capture QC counts per rejection reason, eligibility, source file; per-room summary |
| `manifests/room_B_validation_split.csv` | the one-time Room-B validation split (seed 42, 25 per activity) |
| `manifests/windows.csv` | provenance of every window: capture, start row, split, preprocessing-config hash |
| `windows/development/`, `windows/local_validation/` | `window_XXXXX.npy`, float32 `(150, 256)` linear amplitude |
| `results/results.csv` | `experiment, target_room, K, metric, value` (+ representation, head, routing, direction, std, n_seeds) |
| `results/predictions_validation.csv` | `window_id, room, label, confidence` for every validation window (end-to-end routed system) |
| `results/summary.txt`, `per_class_metrics_target{A,B}_K5.csv` | headline tables; precision/recall/F1 per activity |
| `figures/` | label-efficiency curve, encoder 2-D embedding (UMAP/t-SNE) by room and activity, fingerprint accuracy vs window length, routing contribution, confusion matrices (K=5) + representation ablation |

Experiment names: `task1_room_id`, `task1_fingerprint_vs_length`, `task2_probe_{joint,inroom,cross}_{ssl,random,handcrafted}`,
`task3_{rep}_{head}_{oracle|predicted|none}` (`predicted` = predicted-room routing = headline, `oracle` = true room,
`none` = same adaptation without the routing stage), `task3_end_to_end_routed`. Task-3 metrics: `macro_f1` (primary),
`exclusive_macro_f1` (target-labelled activities), `shared_macro_f1` (transferred SIT/STAND/WALK), `balanced_accuracy`.
`A->B` means source A, target B (target room = `target_room`).

## 5. Design decisions (and why)

**Room A decoding (own NumPy implementation of the public Nexmon/Broadcom format).** UDP port 5500, 18-byte header
(magic `0x1111`, RSSI, frame control, MAC, sequence, core/stream, chanspec, chip) + 256 × 32-bit words; each word packs a
complex value as sign/11-bit mantissa for real and imaginary part plus a 6-bit exponent, followed by a per-packet auto-scale.
The authors' MATLAB code was **not** used, read or ported.

**What the inspection of the data showed** (details in `manifests/qc_summary.json`):
* both rooms share the same centred subcarrier order (guard/DC nulls at the same indices) → no re-ordering;
* Room B CSVs have a header row in every file (`Subcarrier_N` in most, `0..255` in five files) and typically 500 rows;
  file names do not always contain `_Amp_` (e.g. `Empty_11.csv`, `Lying_0.csv`) → the label is the file-name prefix;
* the 80 MHz Room-A data frames alternate with 20 MHz ACK-type frames (`seq == 0xFFFF`), and a further share of the
  data frames is partial-band as well; no Room-A capture even holds 150 non-ACK records (at most ≈ 129), so partial-band
  rows are **flagged, not rejected** (otherwise no official `(150, 256)` window could exist). The normaliser repairs them
  (below) and the QC report counts them;
* a few captures have fewer than 150 valid observations → *ineligible* (listed in `captures.csv`, no window is built).

**QC** (reported per capture): malformed packet (length/magic/not-CSI), secondary stream, wrong CSV width, non-finite,
negative, zero-information (max amplitude < `qc.min_row_peak`), exact duplicate rows.

**Windows**: starts at row 0, stride 35, at most 3 per capture, never across captures, all windows of a capture stay in
its split. **Normalisation** (fitted on development windows only, rooms pooled, no labels): `log1p` → per-row median
removal (gain/AGC differences; the two rooms use different amplitude scales) → time-interpolation of partial-band rows
(`normalization.repair_partial_band_rows`, set `false` to ablate) → per-subcarrier standardisation, clipped.

**Room-B validation split**: 25 captures per activity, `random.Random(42).sample(sorted file names)` per activity, created
once and stored. The challenge text does not name the RNG; change `validation_split.rng_backend` (or `_sample_names()` in
`splits.py`) if the teaching team publishes a reference split — then delete the manifest and re-run from `--stage ingest --force`.

**Task 1** logistic regression on the temporal-mean normalised profile (static multipath fingerprint; length-robust).
**Task 2** 1-D temporal CNN, SimCLR/NT-Xent on development windows of both rooms (no activity or room labels), augmentations
include static-component removal; frozen encoder + linear probe vs. same-architecture random weights vs. handcrafted (81 features).
**Task 3** for each target room T: source room S gives all labelled development windows; T gives K labelled *captures* per
target-exclusive activity (nested draws, 5 seeds); embeddings are standardised per room using unlabelled development windows;
head = balanced logistic regression (primary) or cosine prototypes; routing = Task-1 classifier selects the room-specific model.

## 6. Known limitations (see challenge §12)
Room vs. session confound (the fingerprint may use session effects); activity vs. location confound; two rooms only; capture-disjoint
but not subject-disjoint; ≈ two thirds of Room-A records are partial-band frames, so effective Room-A time resolution is lower than
Room B's; validation numbers are validation results, not unseen-test accuracy.

## 7. AI-use disclosure (complete per course rules)
Parts of this code were drafted with an AI assistant. *Team: describe who reviewed, understood, modified and tested each stage.*

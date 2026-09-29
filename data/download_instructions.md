# Dataset Download Instructions

Both datasets are free and publicly available. Download them and place the files in `data/raw/` before running the pipeline.

---

## 1. NSL-KDD Dataset

**Official source:** University of New Brunswick (UNB)
**URL:** https://www.unb.ca/cic/datasets/nsl.html

### Files needed

| File | Description | Place at |
|---|---|---|
| `KDDTrain+.txt` | Full training set (with difficulty column) | `data/raw/KDDTrain+.txt` |
| `KDDTest+.txt` | Full test set (with difficulty column) | `data/raw/KDDTest+.txt` |
| `KDDTrain+_20Percent.txt` | 20% stratified sample (optional, for fast prototyping) | `data/raw/KDDTrain+_20Percent.txt` |

### Download steps

1. Go to https://www.unb.ca/cic/datasets/nsl.html
2. Scroll to **"Download the Dataset"** and click the NSL-KDD zip link
3. Extract — you will find the `.txt` files listed above
4. Copy them to `data/raw/`

### Column schema (41 features + label + difficulty)

The files have **no header row**. The pipeline adds headers automatically using the known NSL-KDD column order:

```
duration, protocol_type, service, flag, src_bytes, dst_bytes, land,
wrong_fragment, urgent, hot, num_failed_logins, logged_in,
num_compromised, root_shell, su_attempted, num_root, num_file_creations,
num_shells, num_access_files, num_outbound_cmds, is_host_login,
is_guest_login, count, srv_count, serror_rate, srv_serror_rate,
rerror_rate, srv_rerror_rate, same_srv_rate, diff_srv_rate,
srv_diff_host_rate, dst_host_count, dst_host_srv_count,
dst_host_same_srv_rate, dst_host_diff_srv_rate,
dst_host_same_src_port_rate, dst_host_srv_diff_host_rate,
dst_host_serror_rate, dst_host_srv_serror_rate, dst_host_rerror_rate,
dst_host_srv_rerror_rate, label, difficulty
```

---

## 2. UNSW-NB15 Dataset

**Official source:** UNSW Canberra Cyber
**URL:** https://research.unsw.edu.au/projects/unsw-nb15-dataset

### Files needed

| File | Description | Place at |
|---|---|---|
| `UNSW_NB15_training-set.csv` | Training partition | `data/raw/UNSW_NB15_training-set.csv` |
| `UNSW_NB15_testing-set.csv` | Test partition | `data/raw/UNSW_NB15_testing-set.csv` |

### Download steps

1. Go to https://research.unsw.edu.au/projects/unsw-nb15-dataset
2. Find the **"CSV Files"** section and download the training and testing CSVs
3. Alternatively, the dataset is also available on Kaggle:
   - Search for "UNSW-NB15" on https://www.kaggle.com/datasets
   - Direct link (may change): https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15
4. Place the files at the paths shown above

### Column schema

The UNSW-NB15 files include a header row. Key columns used by the pipeline:

- 49 feature columns (numeric + categorical)
- `attack_cat` — the attack category string (mapped via `config/label_mapping.yaml`)
- `label` — binary: 0 = normal, 1 = attack

---

## Setting the active dataset

In your `.env` file (copy from `.env.example`), set:

```
DATASET=nsl_kdd     # or: DATASET=unsw_nb15
```

The pipeline will automatically load the correct files and apply the right mappings.

---

## Verifying the files

After placing files, run:

```bash
python -c "from src.preprocessing import load_raw_data; df = load_raw_data(); print(df.shape)"
```

You should see something like `(125973, 43)` for NSL-KDD full training set.

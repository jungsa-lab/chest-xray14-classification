# Dataset

This project uses the NIH ChestX-ray14 dataset for multi-label chest X-ray classification.

[1] Dataset Information

- Dataset: NIH ChestX-ray14
- Kaggle Dataset: NIH Chest X ray 14 (224x224 resized)
- Author: khanfashee
- Total Images: 112,120
- Unique Patients: 30,805
- Image Size: 224×224
- Number of Disease Labels: 14
- Task: Multi-label Classification

Dataset files are not included in this repository due to their size.

**Do not commit dataset files to this repository.**

[2] Data Structure

Place the downloaded dataset files in the following structure:

```text
data/
├── raw/
│   ├── images/
│   │   ├── 00000001_000.png
│   │   ├── 00000001_001.png
│   │   └── ...
│   └── Data_Entry_2017.csv
├── processed/
└── README.md
```

- `raw/images/` : NIH ChestX-ray14 X-ray images
- `raw/Data_Entry_2017.csv` : Image labels and patient metadata
- `processed/` : Preprocessing output files

[3] Preprocessing

Run the preprocessing script from the project root directory:

```bash
python src/preprocess_data.py
```

The preprocessing pipeline performs:

- Metadata validation
- Missing value and duplicate checks
- Patient age validation
- 14-disease multi-label encoding
- No Finding and label consistency checks
- Image and metadata matching
- Patient-level Train / Validation / Test split
- Patient leakage check
- Image validity check

[4] Preprocessing Output

After preprocessing, the following files are generated in `data/processed/`:

```text
data/processed/
├── preprocessed_metadata.csv
├── split_label_distribution.csv
├── image_validation.csv
├── missing_value_summary.csv
└── preprocessing_report.txt
```

Current preprocessing result:

| Item | Result |
|---|---:|
| Total Images | 112,120 |
| Unique Patients | 30,805 |
| Patient Leakage | 0 |
| Missing Images | 0 |
| Invalid Images | 0 |

[5] Data Split

The dataset is split by **Patient ID** to prevent images from the same patient from appearing in multiple subsets.

| Split | Images | Patients |
|---|---:|---:|
| Train | 78,934 | 21,563 |
| Validation | 11,341 | 3,080 |
| Test | 21,845 | 6,162 |
| Total | 112,120 | 30,805 |

The generated preprocessing files may be used in the following model training and evaluation stages.
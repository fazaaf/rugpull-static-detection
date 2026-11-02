# rugpull-static-detection

Replication package for the manuscript **"Explainable XGBoost for Detecting Rug Pulls in Ethereum-Based DeFi Smart Contracts via Static Code Analysis"** ([citation to be added after publication]).

The study asks whether features computed only from verified source code separate reported rug pull tokens from ordinary tokens on Ethereum. Launch year is matched between classes, the primary analysis is restricted to single-file contracts, near-identical clones are grouped in cross-validation, and test folds contain real contracts only (no synthetic data).

## Repository contents

| Path | Description |
|---|---|
| `data/D_single_era_matched.csv` | Primary dataset: 2,278 single-file ERC-20 contracts (1,864 rug pull, 414 legitimate). Columns: `address`, `label` (1 = rug pull, 0 = legitimate), `source_kind`, `is_proxy` (1 = implementation contract of a proxy was analysed), `year` (launch year, from the first recorded token transfer), `solc_minor` (minor version of the Solidity compiler), and the 29 features. |
| `data/D_main_era_matched.csv` | Secondary dataset with the same columns, including multi-file contracts (1,901 rug pull, 817 legitimate). |
| `data/addresses_rugpull_and_pilot_2475.csv` | Addresses of the rug pull class (2,372, `label` = 1) and of an earlier pilot set of 103 established tokens (`label` = 0). The pilot set was used only for design decisions. |
| `data/legit_candidates_processed_886.csv` | The 886 candidate legitimate tokens that were processed (all tokens launched in 2021-2022 plus a random sample of 25 each from 2020 and 2023). Columns: `address`, `year`, `first_ts`, `last_ts`, `life_days`. |
| `data/cand_span.csv` | First and last recorded token transfer (Unix time, seconds) of the CoinGecko candidate tokens. `last_ts` was collected only for tokens whose first transfer fell in 2020-2023. |
| `data/first_transfer.csv` | First recorded token transfer of the contracts used in the initial analysis (rug pull class and pilot set). |
| `docs/feature_definitions.csv` | Definition of the 29 features, including every regular expression. |
| `notebooks/pipeline_final.ipynb` | Data construction and analyses, with outputs removed. |
| `results/oof_xgb_final.csv` | Out-of-fold XGBoost predictions for the primary dataset. `p_rug` is the mean predicted probability over five cross-validation repetitions, `pred` = 1 if `p_rug` >= 0.5, `life_days` is available for legitimate tokens. |
| `results/shap_importance_final.csv` | Mean absolute SHAP value per feature (log-odds, out-of-fold) and the correlation between feature value and SHAP value (`arah_korelasi`; positive = toward rug pull). |
| `results/versions.txt` | Versions of the Python packages. |

## Labels

- **Rug pull**: contract listed in the rug pull dataset of Sun et al. (2024) or in the ground truth of CRPWarner (Lin et al., 2024).
- **Legitimate**: token from the CoinGecko token list (retrieved October 2026) that appears in neither source and shows at least 365 days between its first and last recorded transfer. This does **not** mean the token is verified safe.

The rug pull class has 2,372 addresses: 2,302 listed in the Ethereum entries of the dataset of Sun et al. (2024), 68 listed in the CRPWarner ground truth (one address appears in both), and 3 further addresses whose origin is still being documented.

## Software

Python 3.13.18; pandas 2.2.3; NumPy 2.1.3; scikit-learn 1.6.1; XGBoost 3.4.1; SHAP 0.54.1; Slither 0.11.6 (run with the compiler version declared on Etherscan). See `results/versions.txt`.

## Using the data

```python
import pandas as pd
df = pd.read_csv("data/D_single_era_matched.csv")
print(df["label"].value_counts())   # 1 = rug pull (1,864), 0 = legitimate (414)
```

## Re-running the pipeline

`notebooks/pipeline_final.ipynb` documents the pipeline as it was run. It expects Google Colab with Google Drive mounted at `/content/drive/MyDrive/rugpull_research`, and retrieving source code requires your own Etherscan API key (replace the placeholder `ISI_API_KEY_ETHERSCAN_ANDA`; never commit a real key). Contract source code is not redistributed here; it can be retrieved from the addresses via Etherscan. The CoinGecko token list changes over time, so the candidate lists in `data/` are the ones used in the study.

## Scope and limitations

- Single-file ERC-20 contracts with verified source code, launched in 2020-2023.
- The rug pull class is dominated by two types (Transaction Limitation 89.8%, Fee 8.3%, per the type labels of Sun et al., 2024), and about 85% of the 1,864 rug pull contracts share a feature signature with another one (295 distinct signatures).
- Legitimate labels mean "no report and sustained activity", not verified safety.
- Performance under a temporal split (training on 2020-2021, testing on 2022-2023) is much lower than under cross-validation.
- The results are not evidence that the model is ready for deployment.

## Licence and attribution

Code and derived data created by the authors are released under the MIT Licence (see `LICENSE`). Rug pull labels derive from Sun et al. (2024) and Lin et al. (2024); legitimate candidates were identified with the CoinGecko token list. Please cite the original sources and check their terms.

## Contact

## Contact

Faza Aulia Fadjri  
Department of Information Technology, State Islamic University Walisongo Semarang  
Email: 23080960010@walisongo.ac.id

# rugpull-static-detection

Replication package for the manuscript **"A Leakage-Aware Evaluation of Static Code Features for Explainable Rug Pull Detection in Ethereum Tokens"** (submitted to VOICE: Journal of Vocational, Informatics and Computer Education; citation to be added after publication). Earlier versions of the manuscript were titled "Explainable XGBoost for Detecting Rug Pulls in Ethereum-Based DeFi Smart Contracts via Static Code Analysis".

The study asks how well features computed only from verified source code separate reported rug pull tokens from ordinary tokens on Ethereum once launch year, compiler, file layout and cloning are controlled. With contracts grouped by identical feature values, XGBoost reaches an MCC of 0.840; when contracts with similar source code are kept on one side of the split, MCC falls to 0.458, and on tokens launched after the training period to 0.412. High scores mainly reflect the recognition of known contract templates.

## Repository contents

| Path | Description |
|---|---|
| `data/D_single_era_matched.csv` | Primary dataset: 2,278 single-file ERC-20 contracts (1,864 rug pull, 414 legitimate). Columns: `address`, `label` (1 = rug pull), `source_kind`, `is_proxy`, `year` (launch year from the first token transfer), `solc_minor`, and the 29 features. |
| `data/D_main_era_matched.csv` | Secondary dataset including multi-file contracts (1,901 rug pull, 818 legitimate). |
| `data/groups.csv` | Grouping used in cross-validation, one row per contract of the primary dataset: `feature_cluster` (706 feature-space clusters), `hash_L1`/`hash_L2` (SHA-256 of the source after removing comments and whitespace / also replacing literals), and `sim_group_0.9`, `sim_group_0.8`, `sim_group_0.7` (MinHash source-similarity groups merged with the feature clusters; 350, 271 and 231 groups). |
| `data/label_sources.csv` | Origin of each label: `source_group` (Trapdoor study via Sun et al., security-firm reports via Sun et al., CRPWarner only, or legitimate), the type label of Sun et al. (`sun_type`), and CRPWarner membership and flag. |
| `data/addresses_rugpull_and_pilot_2475.csv` | Addresses of the rug pull class (2,372, `label` = 1) and of an earlier pilot set of 103 established tokens (`label` = 0), used only for design decisions. |
| `data/legit_candidates_processed_886.csv`, `data/cand_span.csv`, `data/first_transfer.csv` | Legitimate candidates and token-transfer dates (see the column descriptions in earlier releases; unchanged). |
| `docs/feature_definitions.csv` | Definition of the 29 features, including every regular expression. |
| `notebooks/pipeline_final.ipynb` | Original data construction and analyses (Colab), outputs removed. |
| `notebooks/rev2_analyses.ipynb`, `notebooks/rev2lib.py` | Revision analyses run in Colab: re-running Slither with full detector output, detector composition and compiler dependence, enriched metadata baseline, exact duplicates and source-similarity grouping, regex variants V0-V4, activity thresholds, the audit evidence, and the export of `groups.csv`. Outputs removed. `rev2lib.py` holds the helper functions; those marked VERBATIM are copied unchanged from `pipeline_final.ipynb`. |
| `scripts/` | Revision analyses that run on the released data without Colab or an API key (`s1_cv.py` cross-validation and in-fold thresholds, `s2_analyses.py` label sources, compiler strata, calibration, descriptives, activity thresholds, `s3_shap.py` SHAP and stability, `s4_local.py` type sensitivity, random-forest SHAP, SHAP by compiler, perturbation, `s5_collect.py` collects the outputs). |
| `results/oof_xgb_final.csv`, `results/shap_importance_final.csv`, `results/versions.txt` | Outputs of the original pipeline (`arah_korelasi` = correlation between feature value and SHAP value; positive = toward rug pull). |
| `results/rev2/` | Outputs of the revision: `rev2_colab_results.xlsx` (from the Colab notebook), `rev2_analyses.xlsx` and CSV files (from `scripts/`), and the calibration and SHAP dependence figures. |
| `annotation/` | Manual validation of the regular-expression indicators: `regex_annotation_sheet.xlsx` (tabs `codebook`, `annotate`, `rule_decisions` in English and `keputusan_aturan`, the original Indonesian record), `annotation_key.csv` (labels and regex output V0-V3 for the 61 contracts; C01 was a training example), `regex_V4_flags.csv`, `contracts/` (verified source of the 61 annotated contracts as published on Etherscan), `score_annotation.py`, and its output `annotation_validation_results.xlsx`. |
| `audit/top20_legit_audit.xlsx` | Audit of the 20 legitimate tokens with the highest out-of-fold scores: evidence (Etherscan transfers, honeypot.is simulation, GoPlus flags, CoinGecko, web search), the rating rule fixed in advance, ratings, and the first author's check. |
| `teaching/` | Teaching case in three notebooks (see `teaching/README.md`). |

## Labels

- **Rug pull**: contract listed in the rug pull dataset of Sun et al. (2025) or in the ground truth of CRPWarner (Lin et al., 2024).
- **Legitimate**: token from the CoinGecko token list (retrieved October 2026) that appears in neither source and shows at least 365 days between its first and last recorded transfer. This does **not** mean the token is verified safe.

The address list of the rug pull class has 2,372 entries: 2,302 of the 2,303 Ethereum entries of Sun et al. (2025) and 68 of the 69 CRPWarner ground-truth addresses (one address is in both). The remaining Sun et al. entry (0x14aa9c36…) was not in the list we processed, and the remaining CRPWarner address (0x93023f1d…) is recorded by Sun et al. as a BNB Smart Chain token. Three further addresses complete the list: one (0x8ed9c7e4…) is recorded by Sun et al. as a BNB Smart Chain token, and the origin of two (0x5946fe65…, 0x0b1ff525…) was not documented. All three lacked verified source code on Ethereum and were excluded at the first filtering step.

## Reproducing the results

1. **Without Colab or an API key** (about two minutes on a laptop):
   ```bash
   pip install -r requirements.txt
   python scripts/s1_cv.py && python scripts/s2_analyses.py && python scripts/s3_shap.py && python scripts/s4_local.py && python scripts/s5_collect.py
   python annotation/score_annotation.py
   ```
   `s1_cv.py` checks that its out-of-fold predictions match `results/oof_xgb_final.csv` (maximum absolute difference below 1e-6).
2. **Analyses that need contract source code** (source-similarity grouping, Slither detector output, regex variants, metadata baseline, audit) are in `notebooks/rev2_analyses.ipynb`. It expects Google Colab with Google Drive mounted at `/content/drive/MyDrive/rugpull_research`, `rev2lib.py` in `rugpull_research/rev2/`, and an Etherscan API key stored as the Colab secret `ETHERSCAN_API_KEY` (never paste a key into a cell or commit it). Run the numbered sections in order; cells 5f and 5g belong after Section 5, and cell 11 exports `data/groups.csv`.

Contract source code is otherwise not redistributed; it can be retrieved from the addresses via Etherscan. The CoinGecko token list changes over time, so the candidate lists in `data/` are the ones used in the study.

## Software

Python 3.13; pandas 2.2.3; NumPy 2.1.3; scikit-learn 1.6.1; XGBoost 3.4.1; SHAP 0.54.1; Slither 0.11.6 (run with the compiler version declared on Etherscan). See `requirements.txt` and `results/versions.txt`.

## Scope and limitations

- Single-file ERC-20 contracts with verified source code, launched in 2020-2023; compiler versions are not matched between classes.
- The rug pull class is dominated by Trapdoor tokens of two types (Transaction Limitation 89.8%, Fee 8.3%), and one family of near-identical contracts holds 1,190 contracts.
- The regular-expression indicators miss most of the mechanisms they are named after (recall 0.03-0.64 for blacklists, transfer locks, transaction limits and fee changes); the annotation rests on one annotator.
- Legitimate labels mean "no report and sustained activity", not verified safety.
- The results are not evidence that the model is ready for deployment.

## Licence and attribution

Code and derived data created by the authors are released under the MIT Licence (see `LICENSE`). Rug pull labels derive from Sun et al. (2025) and Lin et al. (2024); legitimate candidates were identified with the CoinGecko token list. The files in `annotation/contracts/` are verified source code published on Etherscan by the contract deployers and remain under the licences stated in their SPDX headers; they are included only to make the annotation reproducible. Please cite the original sources and check their terms.

- Sun, D., Ma, W., Nie, L., & Liu, Y. (2025). SoK: A taxonomic analysis of DeFi rug pulls: Types, dataset, and tool assessment. *Proceedings of the ACM on Software Engineering, 2*(ISSTA), 550-572. https://doi.org/10.1145/3728900
- Lin, Z., Chen, J., Wu, J., Zhang, W., Wang, Y., & Zheng, Z. (2024). CRPWarner: Warning the risk of contract-related rug pull in DeFi smart contracts. *IEEE Transactions on Software Engineering, 50*(6), 1534-1547. https://doi.org/10.1109/TSE.2024.3392451

## Contact

Faza Aulia Fadjri  
Department of Information Technology, State Islamic University Walisongo Semarang  
Email: 23080960010@student.walisongo.ac.id

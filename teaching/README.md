# Teaching case: leakage, indicator validity and explanations in rug pull detection

A three-session case for undergraduate informatics or vocational students who have completed an introductory machine-learning course. Each session takes about 100 minutes and uses only the data in this repository (no API key, no blockchain access).

| Session | Notebook | Learning objectives |
|---|---|---|
| 1 | `session1_leakage_and_splits.ipynb` | (1) explain how cloned contracts inflate cross-validated scores; (3) compare random, feature-grouped, similarity-grouped and temporal splits |
| 2 | `session2_validating_indicators.ipynb` | (2) read contract code and static-analysis output; (5) validate a pattern-based indicator against manual inspection |
| 3 | `session3_explanations_and_evasion.ipynb` | (4) distinguish what a SHAP explanation says about a model from what it says about a fraud mechanism; evaluate evasion; write a one-page critique of a screening claim |

**Running.** Locally: `pip install -r requirements.txt`, then open the notebooks from this folder. In Google Colab: open a notebook from GitHub; the first cell clones the repository and installs the packages. Sessions 1 and 3 run in under a minute on a laptop.

**Materials.** `data/D_single_era_matched.csv` (29 features, labels), `data/groups.csv` (feature clusters, source fingerprints, similarity groups), `annotation/` (codebook, rule decisions, the authors' annotation of 60 contracts, contract sources), `audit/` (audit of the 20 highest-scoring legitimate tokens).

**For instructors.** Session 2 asks students to annotate five contracts (C53, C39, C51, C22, C61) before they open the authors' annotation; hand out the codebook first. C61 is a legitimate token whose code is identical to a reported rug pull template, and C22 is a reported rug pull without any of the coded mechanisms; both are good discussion starters. Numbers in the notebooks use one cross-validation repetition and differ slightly from the paper, which averages five.

**Status.** The case has not yet been used in a course, and no learning outcomes have been measured.

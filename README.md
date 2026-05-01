# Mental Health in the Tech Industry

CS 210: Data Management for Data Science  
Authors: Vanij Patel and Sean Lumasag

## Project Overview

This project analyzes mental health treatment-seeking behavior among technology workers using the Open Sourcing Mental Illness (OSMI) Mental Health in Tech surveys from 2014 and 2016.

Research question:

> What workplace and personal factors predict whether a tech worker will seek mental health treatment, and how does mental health interfere with their work?

The project combines two survey datasets with different schemas, cleans and aligns their columns, stores the processed data in SQLite, runs SQL-based analysis, engineers workplace support and stigma features, and trains machine learning models to predict treatment-seeking behavior.

## Repository Contents

| File or Folder | Description |
| --- | --- |
| `notebook.ipynb` | Main final notebook. Run this from top to bottom to reproduce the project. |
| `2014.csv` | OSMI 2014 Mental Health in Tech survey data. |
| `2016.csv` | OSMI 2016 Mental Health in Tech survey data. |
| `mental_health.db` | SQLite database created by the notebook. Includes raw and processed project tables. |
| `output_figures/` | Generated visualizations used in the analysis and report. |
| `report.pdf` | Final written project report. |
| `proposal.pdf` | Original project proposal. |
| `requirements.txt` | Python dependencies required to run the notebook. |

## Data

The project uses two public OSMI survey datasets:

| Dataset | Year | Rows | Columns |
| --- | ---: | ---: | ---: |
| OSMI Mental Health in Tech Survey | 2014 | 1,259 | 27 |
| OSMI Mental Health in Tech Survey | 2016 | 1,433 | 63 |
| Merged aligned dataset | 2014 + 2016 | 2,692 | 26 |

The 2014 survey uses short column names such as `self_employed`, `work_interfere`, and `no_employees`. The 2016 survey uses full survey-question text as column names. A major part of the project is manually mapping the 2016 question columns to the 2014 schema so both years can be analyzed together.

## Methodology

The main notebook follows this pipeline:

1. Load the 2014 and 2016 CSV files into Pandas.
2. Store the raw datasets in `mental_health.db`.
3. Inspect null rates, data types, and column differences.
4. Manually align the 2016 survey schema to the 2014 schema.
5. Clean invalid ages, standardize gender values, normalize binary columns, and handle missing `work_interfere` values.
6. Merge both surveys into one aligned dataset with a `survey_year` column.
7. Engineer two composite features:
   - `employer_support_score`
   - `stigma_score`
8. Run analytical SQL queries against the processed SQLite table.
9. Train and evaluate Logistic Regression and Random Forest models.
10. Generate visualizations and save them to `output_figures/`.

## Key Results

The final merged dataset contains 2,692 respondents: 1,259 from 2014 and 1,433 from 2016.

Treatment-seeking increased between the two survey years:

| Year | Treatment Rate |
| ---: | ---: |
| 2014 | 50.6% |
| 2016 | 58.5% |

Treatment-seeking was strongly associated with how often mental health interfered with work:

| Work Interference Level | Treatment Rate |
| --- | ---: |
| Often | 88.1% |
| Sometimes | 73.8% |
| Rarely | 67.6% |
| Never | 15.6% |
| Not Applicable | 8.8% |

Model performance for predicting treatment-seeking:

| Model | Accuracy | F1 | ROC-AUC |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.738 | 0.761 | 0.795 |
| Random Forest | 0.759 | 0.782 | 0.827 |

The Random Forest model performed best overall. Workplace-related features were slightly stronger predictors of treatment-seeking than personal features:

| Feature Group | ROC-AUC |
| --- | ---: |
| Workplace features | 0.748 |
| Personal features | 0.732 |

The `survey_year` feature ranked 8th out of 23 features in the Random Forest feature-importance analysis, meaning year-to-year differences were moderately predictive but not dominant.

## Visualizations

The notebook generates the following visualizations in `output_figures/`:

| Figure | File |
| --- | --- |
| Treatment-seeking rate by country | `vis1_treatment_by_country.png` |
| Year-over-year treatment, employer support, and stigma comparison | `vis2_year_over_year.png` |
| Treatment rate by employer support score bin | `vis3_employer_support_bins.png` |
| Employer support scatter plot | `vis3_employer_support_scatter.png` |
| Workplace stigma score vs treatment avoidance | `vis4_stigma_vs_treatment.png` |
| Willingness to discuss mental health by relationship type | `vis5_discuss_by_relationship.png` |
| Top Random Forest feature importances | `vis6_feature_importance.png` |
| Model performance comparison | `vis7_model_comparison.png` |
| Confusion matrices | `vis8_confusion_matrices.png` |
| Personal vs workplace AUC comparison | `vis9_personal_vs_workplace_auc.png` |
| Age distribution by treatment status | `vis10_age_distribution.png` |

## How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Open `notebook.ipynb` and run all cells from top to bottom.

The notebook expects the following files to be present in the repository root:

- `2014.csv`
- `2016.csv`

Running the notebook will create or update:

- `mental_health.db`
- generated figures in `output_figures/`

## Requirements

The project uses:

- Python
- Jupyter Lab
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- SQLite

See `requirements.txt` for the installable Python package list.

## Demo Video

TBD
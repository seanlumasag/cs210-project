# Mental Health in the Tech Industry

CS 210: Data Management for Data Science  
Authors: Vanij Patel and Sean Lumasag

## Project Overview

This project analyzes mental health treatment-seeking behavior among technology workers using the Open Sourcing Mental Illness (OSMI) Mental Health in Tech surveys from 2014 and 2016.

Research question:

> What workplace and personal factors predict whether a tech worker will seek mental health treatment, and how does mental health interfere with their work?

The project combines two survey datasets with different schemas, cleans and aligns their columns, stores the processed data in SQLite, runs SQL-based analysis, engineers workplace support and stigma features, and trains machine learning models to predict treatment-seeking behavior.

## How to Run

These instructions are for running the project on a Mac.

### 1. Install Python

For reproducible notebook metrics, use **Python 3.14.3**. Other Python versions, including Python 3.14.2, may produce slightly different machine learning metrics even when the notebook code is unchanged.

If Python 3.14.3 is not already installed, download and install it from:

https://www.python.org/downloads/

After installing Python, open the Terminal app and check that the correct Python version is available:

```bash
python3.14 --version
```

You should see:

```text
Python 3.14.3
```

### 2. Clone the Repository

In Terminal, move into the folder where you want to save the project, then clone the repository:

```bash
git clone https://github.com/seanlumasag/cs210-project.git
```

### 3. Open the Project Folder in Terminal

Move into the cloned project folder:

```bash
cd cs210-project
```

If you cloned the repository somewhere else, replace the path after `cd` with the location of your project folder.

### 4. Create the Project Environment and Install Packages

Run these commands from inside the project folder:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name cs210-python-3.14.3 --display-name "CS210 Python 3.14.3"
```

This creates a local virtual environment, installs the pinned packages needed for the project, and registers a Jupyter kernel named `CS210 Python 3.14.3`.

### 5. Run the Notebook in VS Code

Open the cloned `cs210-project` folder in VS Code, then open `notebook.ipynb`.

Before running the notebook, make sure these VS Code extensions are installed:

- Python
- Jupyter

VS Code usually prompts you to install these extensions when you open `notebook.ipynb`. If it does not, open the Extensions panel in VS Code and install them manually.

If VS Code asks you to select a kernel or interpreter, choose:

```text
CS210 Python 3.14.3
```

To verify that the notebook is using the correct environment, run this in a notebook cell:

```python
import sys
import pandas as pd
import numpy as np
import sklearn

print(sys.version)
print("pandas", pd.__version__)
print("numpy", np.__version__)
print("sklearn", sklearn.__version__)
```

Expected package versions:

```text
pandas 3.0.1
numpy 2.4.3
sklearn 1.8.0
```

Run the notebook from top to bottom by selecting:

```text
Run All
```

The notebook expects these files to be in the same project folder:

- `notebook.ipynb`
- `2014.csv`
- `2016.csv`

Running all cells will create or update:

- `mental_health.db`
- generated figures in `output_figures/`

Existing versions of those files may be overwritten when the notebook runs.

### Troubleshooting

If `python3` is not found, Python may not be installed correctly. Reinstall Python 3 from python.org, then close and reopen Terminal.

If `python3.14 --version` does not show Python 3.14.3, install Python 3.14.3 and reopen Terminal before continuing.

If you see an error like `No module named pandas` or `No module named sklearn`, activate the project environment and reinstall the requirements:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If model metrics differ between computers, confirm both computers are using Python 3.14.3, the `CS210 Python 3.14.3` notebook kernel, and the pinned package versions from `requirements.txt`. Then restart the kernel and run all cells from top to bottom.

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
| `requirements.txt` | Python packages needed to run the notebook. |

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
| Logistic Regression | 0.737 | 0.759 | 0.795 |
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
| Workplace stigma score vs treatment avoidance | `vis4_stigma_vs_treatment.png` |
| Willingness to discuss mental health by relationship type | `vis5_discuss_by_relationship.png` |
| Top Random Forest feature importances | `vis6_feature_importance.png` |
| Model performance comparison | `vis7_model_comparison.png` |
| Confusion matrices | `vis8_confusion_matrices.png` |
| Personal vs workplace AUC comparison | `vis9_personal_vs_workplace_auc.png` |
| Age distribution by treatment status | `vis10_age_distribution.png` |

## Requirements

The project uses:

- Python 3.14.3
- VS Code with notebook support
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- SQLite

See `requirements.txt` for the installable Python package list. SQLite support comes with Python, so it does not need to be installed separately.

## Demo Video

[Demo Video](https://drive.google.com/file/d/12N-6WGhYzJGnq7YqYggkBM4-CiCeUEhW/view?usp=share_link)

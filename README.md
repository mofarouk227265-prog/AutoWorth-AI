# AutoWorth AI — Used Car Price & Deal Advisor

Individual final project for **Artificial Intelligence – Basic Level B2**.

AutoWorth AI estimates the fair market price of a used car from the 100,000 UK Used Car Dataset, then compares that estimate with the seller's asking price and returns a deal rating.

## What the system answers

> What is the expected market price of this car, and is the seller offering a good deal?

Example:

- Estimated market price: £18,500
- Seller price: £17,000
- Difference: £1,500 cheaper
- Deal rating: Good Deal

## Dataset

Kaggle: [100,000 UK Used Car Dataset](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes)

Manufacturer files used for modelling:

- Audi, BMW, Ford, Hyundai, Mercedes, Skoda, Toyota, Vauxhall, Volkswagen

`cclass.csv` / `focus.csv` and the unclean files are inspected but **not** merged into the modelling table. They are missing `tax` and `mpg`, and C-Class / Focus listings already appear in the Mercedes and Ford files.

## Project pipeline

1. Load each manufacturer file and add a `make` column  
2. Combine into one dataset  
3. Investigate and clean missing values, duplicates, and unrealistic values  
4. Exploratory data analysis (7 plots in `figures/`)  
5. Train / validation / test split: **70% / 15% / 15%** before preprocessing  
6. Feature engineering (car age, mileage per year, premium flag, engine efficiency)  
7. One-hot encoding + scaling fitted on **training data only**  
8. Train five regressors and compare R², MAE, RMSE  
9. Check overfitting (train vs validation)  
10. Randomized search on XGBoost  
11. Evaluate the chosen model **once** on the held-out test set  
12. Actual vs predicted plot, feature importance, large-error inspection  
13. Smart Deal Advisor + Streamlit app  

## Deal ratings

| Rating | Rule |
| --- | --- |
| Great Deal | Seller is more than 10% cheaper than predicted market price |
| Good Deal | 5–10% cheaper |
| Fair Price | Within about ±5% |
| Slightly Overpriced | 5–10% more expensive |
| Overpriced | More than 10% above predicted market price |

## Repository layout

| Path | Purpose |
| --- | --- |
| `AutoWorth_AI.ipynb` | Full walkthrough (use this in Google Colab) |
| `train_and_export.py` | Reproducible training script that writes model artifacts |
| `app.py` | Streamlit Smart Deal Advisor |
| `artifacts/` | Saved model bundle and metrics |
| `figures/` | EDA and model plots |
| `AutoWorth_AI_Presentation.pptx` | Project slides |
| `requirements.txt` | Python dependencies |

## Results from the trained pipeline

Held-out **test** performance for the final **Tuned XGBoost** model:

| Split | R² | MAE | RMSE |
| --- | --- | --- | --- |
| Train | 0.982 | £886 | £1,287 |
| Validation | 0.966 | £1,090 | £1,722 |
| Test | **0.967** | **£1,080** | **£1,677** |

Tuning improved XGBoost validation R² from 0.954 to 0.966. The train–validation gap is small, so the model is not just memorising the training listings. The assignment target of R² > 0.90 is met.

## How to run locally

```bash
python -m pip install -r requirements.txt
python train_and_export.py
streamlit run app.py
```

Open `AutoWorth_AI.ipynb` in Jupyter or upload it to Google Colab with the CSV files in the same folder.

## Design choices worth defending

- **Reference year = 2020**, because that is the latest listing year in this dump. Car age is `2020 - year`.
- **Mileage per year** uses `max(car_age, 1)` so brand-new (age 0) cars do not divide by zero.
- **Premium flag** (`Audi` / `BMW` / `Mercedes`) captures brand positioning that one-hot encoding also sees, but as a compact numeric signal for linear models.
- **Engine efficiency** (`mpg / engineSize`) describes how far the car travels per litre of capacity.
- Preprocessing is fitted on the training split only to avoid leakage.
- Hyperparameter search uses a training subsample, then the winning settings are refit on the full training set. The test set stays untouched until the final model is chosen.

## Author

Individual coursework project. All analysis, cleaning decisions, models, and write-up are original work for this submission.

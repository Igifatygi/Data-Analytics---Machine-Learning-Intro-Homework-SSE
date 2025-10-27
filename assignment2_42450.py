"""Comprehensive housing market modelling pipeline.

The original project shipped as a long, monolithic script that relied on a
hard-coded working directory and produced a large number of chained
assignments that triggered Pandas warnings.  This refactored version keeps the
original modelling logic but organises it into clear, reusable functions and
annotates every major step with comments so that new collaborators can follow
what is happening without having to reverse-engineer the code flow.

The pipeline performs the following high-level steps:
1. Load the three source tables either from a SQL database or from CSV files.
2. Clean the annual report and housing association data.
3. Engineer apartment level features and merge the auxiliary datasets.
4. Impute missing room counts and rents with tuned XGBoost models.
5. Train multiple models (Linear Regression, KNN, Decision Tree, XGBoost) for
   explanatory purposes and tune the best performer (XGBoost).
6. Use the tuned XGBoost model to score the validation subset and export the
   predictions as JSON.

Every helper returns a copy of the data it receives which keeps the call chain
side-effect free.  Logging is added so the script can be followed when executed
in batch mode.  The script still assumes that the underlying schema matches the
original homework specification, but the code is now easier to adapt to other
sources.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import logging
import os

import numpy as np
import pandas as pd
import sqlalchemy as sql
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

@dataclass
class DataSource:
    """Stores configuration for a table location."""

    table: str
    index_col: str
    csv_path: Optional[Path] = None


@dataclass
class DatabaseCredentials:
    """Stores the credentials required to connect to the SQL database."""

    host: str
    username: str
    password: str
    schema: str

    def to_sqlalchemy_url(self) -> str:
        """Compose a SQLAlchemy connection string from the credentials."""

        return (
            f"mysql+pymysql://{self.username}:{self.password}@"
            f"{self.host}/{self.schema}"
        )


@dataclass
class ImputationModel:
    """Container that stores the fitted imputer and its feature columns."""

    model: XGBRegressor
    feature_columns: List[str]


# ---------------------------------------------------------------------------
# Loading utilities
# ---------------------------------------------------------------------------

def build_sql_engine(credentials: DatabaseCredentials) -> sql.engine.Engine:
    """Create a SQLAlchemy engine from the provided credentials."""

    logging.debug("Creating SQLAlchemy engine for host %s", credentials.host)
    return sql.create_engine(credentials.to_sqlalchemy_url())


def load_table(
    source: DataSource,
    engine: Optional[sql.engine.Engine] = None,
) -> pd.DataFrame:
    """Load a single table either from CSV or from a SQL engine."""

    if source.csv_path is not None:
        logging.info("Loading %s from CSV at %s", source.table, source.csv_path)
        return pd.read_csv(source.csv_path, index_col=source.index_col)

    if engine is None:
        raise ValueError(
            f"No engine provided for SQL load of table '{source.table}'."
        )

    query = sql.text(f"SELECT * FROM {source.table}")
    logging.info("Loading %s directly from SQL", source.table)
    return pd.read_sql_query(con=engine.connect(), sql=query, index_col=source.index_col)


def load_datasets(
    credentials: Optional[DatabaseCredentials],
    apartment_source: DataSource,
    report_source: DataSource,
    housing_source: DataSource,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three core datasets required by the pipeline."""

    engine = build_sql_engine(credentials) if credentials else None
    apartment_df = load_table(apartment_source, engine)
    report_df = load_table(report_source, engine)
    house_df = load_table(housing_source, engine)
    return apartment_df, report_df, house_df


# ---------------------------------------------------------------------------
# Cleaning routines for the auxiliary tables
# ---------------------------------------------------------------------------

def clean_annual_report(report_df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and aggregate financial metrics per association."""

    report = report_df.copy()

    # Replace problematic codes with NaNs and fill using the documented rules.
    report["total_loan"].replace({1: np.nan, -1: 0}, inplace=True)
    report["total_loan"].fillna(report["total_loan"].mean(), inplace=True)

    categorical_modes = {
        "association_tax_liability": report["association_tax_liability"].mode()[0],
        "long_term_debt_other": report["long_term_debt_other"].mode()[0],
        "number_of_rental_units": report["number_of_rental_units"].mode()[0],
        "total_commercial_area": report["total_commercial_area"].mode()[0],
        "total_plot_area": report["total_plot_area"].mode()[0],
        "total_rental_area": report["total_rental_area"].mode()[0],
    }
    for column, value in categorical_modes.items():
        report[column].fillna(value, inplace=True)

    report["savings"].fillna(report["savings"].mean(), inplace=True)

    # Restrict the columns to the ones used downstream and aggregate per org.
    selected_columns = [
        "association_tax_liability",
        "plot_is_leased",
        "savings",
        "total_commercial_area",
        "total_living_area",
        "total_loan",
        "total_plot_area",
        "total_rental_area",
    ]
    report.reset_index(inplace=True)
    report = report[["org_number", *selected_columns]]

    aggregations = {
        "association_tax_liability": lambda x: x.mode().iloc[0],
        "plot_is_leased": lambda x: x.mode().iloc[0],
        "savings": "mean",
        "total_commercial_area": "mean",
        "total_living_area": "mean",
        "total_loan": "mean",
        "total_plot_area": "mean",
        "total_rental_area": "mean",
    }
    report_grouped = report.groupby("org_number").agg(aggregations)
    logging.debug("Annual report cleaned with %d rows", len(report_grouped))
    return report_grouped


def clean_housing_association(house_df: pd.DataFrame) -> pd.DataFrame:
    """Create period buckets for construction year per association."""

    house = house_df.copy()

    # Fill the missing years with a sentinel and convert to labelled bins.
    house["construction_year"].fillna(-1, inplace=True)
    bins = [-1, 1650, 1923, 1943, 1973, 2003, 2023]
    labels = ["Unknown", "Very Old", "Old", "Medium", "New", "Very New"]
    house["Period"] = pd.cut(
        house["construction_year"], bins=bins, labels=labels, include_lowest=True
    )

    result = house.reset_index()[["org_number", "Period"]]
    logging.debug("Housing association cleaned with %d rows", len(result))
    return result


# ---------------------------------------------------------------------------
# Apartment feature engineering
# ---------------------------------------------------------------------------

def fill_legal_district(apartments: pd.DataFrame) -> pd.Series:
    """Back-fill missing legal district labels using postcode mappings."""

    postcode_map = (
        apartments[["postcode", "legal_district"]]
        .dropna(subset=["legal_district"])
        .drop_duplicates()
        .set_index("postcode")["legal_district"]
    )

    def _impute(row: pd.Series) -> str:
        if pd.notna(row["legal_district"]):
            return row["legal_district"]
        postcode = row["postcode"]
        if pd.notna(postcode) and postcode in postcode_map:
            return postcode_map.loc[postcode]
        return "Unknown"

    return apartments.apply(_impute, axis=1)


def engineer_apartment_features(
    apartment_df: pd.DataFrame,
    house_df: pd.DataFrame,
    report_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create modelling features and validation subset from apartment data."""

    apartments = apartment_df.copy()

    # Convert sell date to year so we can treat it as a numeric feature.
    apartments["sell_date"] = pd.to_datetime(apartments["sell_date"])
    apartments["sell_year"] = apartments["sell_date"].dt.year.astype("Int64")
    apartments.drop(columns=["sell_date"], inplace=True)

    # Standardise the organisation key used for joins.
    apartments.rename(
        columns={"housing_association_org_number": "org_number"}, inplace=True
    )

    # Drop fields that are not used in the downstream models.
    apartments.drop(
        columns=["has_balcony", "width", "height", "energy_class"],
        inplace=True,
        errors="ignore",
    )

    # Replace missing binary/categorical flags with sensible defaults.
    binary_columns = ["has_solar_panels", "has_fireplace", "has_patio"]
    for column in binary_columns:
        apartments[column] = apartments[column].fillna(apartments[column].mode()[0])

    for column in ["additional_area", "plot_area"]:
        apartments[column] = apartments[column].fillna(0)

    # Infer missing legal district values using the helper defined above.
    apartments["legal_district"] = fill_legal_district(apartments)

    # Estimate missing operating costs based on averages per apartment size.
    mean_operating_cost = apartments["operating_cost"].mean()
    mean_cost_per_area = (
        apartments["operating_cost"] / apartments["living_area"]
    ).mean()

    def _fill_operating(row: pd.Series) -> float:
        if pd.notna(row["operating_cost"]):
            return row["operating_cost"]
        if pd.notna(row["living_area"]):
            return float(mean_cost_per_area * row["living_area"])
        return float(mean_operating_cost)

    apartments["operating_cost"] = apartments.apply(_fill_operating, axis=1)

    # Merge auxiliary datasets to enrich the apartment features.
    apartments = apartments.reset_index()
    apartments = apartments.merge(house_df, on="org_number", how="left")
    apartments = apartments.merge(report_df, on="org_number", how="left")
    apartments.set_index("id", inplace=True)

    # Drop free-text and identifier columns that are not predictive.
    drop_columns = [
        "locality",
        "brokers_description",
        "postcode",
        "asking_price",
        "street_name",
        "street_address",
        "org_number",
        "is_new_construction",
    ]
    apartments.drop(columns=[c for c in drop_columns if c in apartments.columns], inplace=True)

    # Cast categorical columns to pandas categorical dtype for clarity.
    categorical_columns = [
        "legal_district",
        "object_type",
        "Period",
        "association_tax_liability",
    ]
    for column in categorical_columns:
        apartments[column] = apartments[column].astype("category")

    # Split into clean modelling set and validation set (where sell_price is NaN).
    validation_set = apartments[apartments["sell_price"].isna()].copy()
    modelling_set = apartments.dropna(subset=["sell_price"]).copy()

    logging.info(
        "Prepared apartment features with %d modelling rows and %d validation rows",
        len(modelling_set),
        len(validation_set),
    )
    return modelling_set, validation_set


# ---------------------------------------------------------------------------
# Imputation helpers
# ---------------------------------------------------------------------------

def tune_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    param_grid: Dict[str, Sequence],
) -> XGBRegressor:
    """Tune an XGBoost regressor and return the best estimator."""

    model = XGBRegressor(objective="reg:squarederror", enable_categorical=True)
    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=3,
        n_jobs=-1,
    )
    search.fit(X, y)
    logging.info("Best XGBoost params: %s", search.best_params_)
    return search.best_estimator_


def impute_with_model(
    full_df: pd.DataFrame,
    target_column: str,
    drop_columns: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, ImputationModel]:
    """Impute missing values in `target_column` using a tuned XGBoost model."""

    drop_columns = set(drop_columns or [])
    drop_columns.update({target_column, "sell_price"})

    # Prepare the clean subset used for fitting the imputation model.
    features = full_df.drop(columns=list(drop_columns), errors="ignore")
    target = full_df[target_column]
    clean_mask = target.notna()
    X_train = features.loc[clean_mask]
    y_train = target.loc[clean_mask]

    if X_train.empty:
        raise ValueError(
            f"No non-null rows available to train imputer for column '{target_column}'."
        )

    # Apply the standard parameter grid used throughout the project.
    grid = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 4, 5],
        "min_child_weight": [1, 2, 3],
    }
    best_model = tune_xgboost(X_train, y_train, grid)

    # Predict missing values and fill them in the original frame copy.
    result = full_df.copy()
    missing_mask = result[target_column].isna()
    if missing_mask.any():
        predictions = best_model.predict(features.loc[missing_mask])
        result.loc[missing_mask, target_column] = predictions
        logging.info(
            "Imputed %d values for column %s", missing_mask.sum(), target_column
        )

    imputer = ImputationModel(best_model, list(features.columns))
    return result, imputer


def apply_imputer_model(
    df: pd.DataFrame,
    target_column: str,
    imputer: ImputationModel,
    drop_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Apply a previously fitted imputation model to another dataset."""

    result = df.copy()
    drop_columns = set(drop_columns or [])
    drop_columns.update({target_column, "sell_price"})

    features = result.drop(columns=list(drop_columns), errors="ignore")
    for column in imputer.feature_columns:
        if column not in features.columns:
            features[column] = 0
    features = features[imputer.feature_columns]

    missing_mask = result[target_column].isna()
    if missing_mask.any():
        predictions = imputer.model.predict(features.loc[missing_mask])
        result.loc[missing_mask, target_column] = predictions
        logging.info(
            "Applied imputer for column %s to %d rows", target_column, missing_mask.sum()
        )
    return result


def prepare_training_matrices(
    modelling_df: pd.DataFrame,
    categorical_columns: Sequence[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    """Encode categorical columns and separate features from the target."""

    encoded = pd.get_dummies(modelling_df, columns=list(categorical_columns), prefix="Category")
    X = encoded.drop(columns=["sell_price"])
    y = encoded["sell_price"]
    return X, y


# ---------------------------------------------------------------------------
# Model training and evaluation
# ---------------------------------------------------------------------------

def evaluate_models(X: pd.DataFrame, y: pd.Series, kfold: KFold) -> Dict[str, Dict[str, float]]:
    """Train a suite of baseline models and report their cross-validated scores."""

    scoring = {"r2": "r2", "rmse": "neg_root_mean_squared_error"}
    results: Dict[str, Dict[str, float]] = {}

    # Linear Regression baseline.
    linear_model = LinearRegression()
    linear_scores = cross_validate(linear_model, X, y, cv=kfold, scoring=scoring)
    results["linear_regression"] = {metric: scores.mean() for metric, scores in linear_scores.items()}

    # KNN baseline across different neighbour counts.
    best_knn: Tuple[int, float] = (0, float("-inf"))
    for neighbours in range(3, 8):
        knn = KNeighborsRegressor(n_neighbors=neighbours)
        scores = cross_validate(knn, X, y, cv=kfold, scoring=scoring)
        mean_r2 = scores["test_r2"].mean()
        results[f"knn_{neighbours}"] = {metric: values.mean() for metric, values in scores.items()}
        if mean_r2 > best_knn[1]:
            best_knn = (neighbours, mean_r2)
    logging.info("Best KNN neighbour count: %d", best_knn[0])

    # Decision Tree baseline with tuning.
    tree = DecisionTreeRegressor(random_state=0)
    tree_grid = {
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    tree_search = GridSearchCV(tree, tree_grid, cv=kfold, scoring="r2", n_jobs=-1)
    tree_search.fit(X, y)
    best_tree = tree_search.best_estimator_
    tree_scores = cross_validate(best_tree, X, y, cv=kfold, scoring=scoring, return_train_score=True)
    results["decision_tree"] = {metric: values.mean() for metric, values in tree_scores.items()}

    # XGBoost baseline with tuning.
    xgboost_grid = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 4, 5],
        "min_child_weight": [1, 2, 3],
    }
    best_xgb = tune_xgboost(X, y, xgboost_grid)
    xgb_scores = cross_validate(best_xgb, X, y, cv=kfold, scoring=scoring, return_train_score=True)
    results["xgboost"] = {metric: values.mean() for metric, values in xgb_scores.items()}

    return results


def assess_robustness(
    encoded_df: pd.DataFrame,
    xgb_model: XGBRegressor,
    kfold: KFold,
    scoring: Dict[str, str],
    random_state: int = 42,
) -> np.ndarray:
    """Remove random columns and evaluate robustness of the tuned model."""

    rng = np.random.default_rng(random_state)
    feature_columns = [col for col in encoded_df.columns if col != "sell_price"]
    drop_count = rng.integers(1, min(4, len(feature_columns)))
    drop_columns = rng.choice(feature_columns, size=drop_count, replace=False)
    logging.info("Testing robustness by dropping columns: %s", ", ".join(drop_columns))

    reduced_df = encoded_df.drop(columns=drop_columns)
    X = reduced_df.drop(columns=["sell_price"])
    y = reduced_df["sell_price"]
    scores = cross_validate(xgb_model, X, y, cv=kfold, scoring=scoring, return_train_score=True)
    return scores["test_r2"]


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def predict_validation_prices(
    validation_df: pd.DataFrame,
    xgb_model: XGBRegressor,
    categorical_columns: Sequence[str],
    training_features: Sequence[str],
    output_path: Path,
) -> None:
    """Score the validation subset and export the predictions as JSON."""

    encoded_validation = pd.get_dummies(validation_df, columns=list(categorical_columns), prefix="Category")
    X_validation = encoded_validation.drop(columns=["sell_price"], errors="ignore")
    for column in training_features:
        if column not in X_validation.columns:
            X_validation[column] = 0
    X_validation = X_validation[training_features]
    predictions = xgb_model.predict(X_validation)

    payload = pd.DataFrame({
        "id": encoded_validation.index,
        "sell_price": predictions,
    })
    payload.to_json(output_path, orient="records", indent=2)
    logging.info("Validation predictions exported to %s", output_path)


# ---------------------------------------------------------------------------
# Main execution flow
# ---------------------------------------------------------------------------

def main() -> None:
    """Execute the full modelling pipeline."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Step 1: Build configuration from environment variables so the script can
    # run without hard-coded secrets in source control.
    credentials = None
    if os.environ.get("DB_HOST"):
        credentials = DatabaseCredentials(
            host=os.environ["DB_HOST"],
            username=os.environ["DB_USERNAME"],
            password=os.environ["DB_PASSWORD"],
            schema=os.environ["DB_SCHEMA"],
        )

    apartment_csv = os.environ.get("APARTMENT_CSV")
    report_csv = os.environ.get("ANNUAL_REPORT_CSV")
    housing_csv = os.environ.get("HOUSING_ASSOCIATION_CSV")

    apartment_source = DataSource(
        "Apartment", index_col="id", csv_path=Path(apartment_csv) if apartment_csv else None
    )
    report_source = DataSource(
        "AnnualReport",
        index_col="org_number",
        csv_path=Path(report_csv) if report_csv else None,
    )
    housing_source = DataSource(
        "HousingAssociation",
        index_col="org_number",
        csv_path=Path(housing_csv) if housing_csv else None,
    )

    apartment_df, report_df, housing_df = load_datasets(
        credentials, apartment_source, report_source, housing_source
    )

    # Step 2: Clean the auxiliary datasets.
    cleaned_report = clean_annual_report(report_df)
    cleaned_housing = clean_housing_association(housing_df)

    # Step 3: Engineer apartment features and split modelling vs validation sets.
    modelling_df, validation_df = engineer_apartment_features(
        apartment_df, cleaned_housing, cleaned_report
    )

    # Step 4: Impute missing intermediary targets (rooms and rent) using tuned models.
    modelling_df, rooms_imputer = impute_with_model(
        modelling_df, "rooms", drop_columns=["rent"]
    )
    modelling_df, rent_imputer = impute_with_model(modelling_df, "rent")
    validation_df = apply_imputer_model(
        validation_df, "rooms", rooms_imputer, drop_columns=["rent"]
    )
    validation_df = apply_imputer_model(validation_df, "rent", rent_imputer)

    # Step 5: Prepare matrices for model training.
    categorical_columns = ["legal_district", "object_type", "Period", "association_tax_liability"]
    X, y = prepare_training_matrices(modelling_df, categorical_columns)
    training_features = list(X.columns)

    # Step 6: Evaluate multiple models and keep the tuned XGBoost for deployment.
    kfold = KFold(n_splits=5, shuffle=True, random_state=80)
    evaluation_results = evaluate_models(X, y, kfold)
    logging.info("Model evaluation summary: %s", evaluation_results)

    tuned_xgb = tune_xgboost(X, y, {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 4, 5],
        "min_child_weight": [1, 2, 3],
    })

    # Step 7: Assess the robustness of the tuned model by dropping random features.
    encoded_full = pd.concat([X, y], axis=1)
    scoring = {"r2": "r2", "rmse": "neg_root_mean_squared_error"}
    robustness_scores = assess_robustness(encoded_full, tuned_xgb, kfold, scoring)
    logging.info("Robustness test R2 scores: %s", robustness_scores)

    # Step 8: Score the validation subset and export the predictions.
    output_path = Path("assignment2_42441.json")
    predict_validation_prices(
        validation_df, tuned_xgb, categorical_columns, training_features, output_path
    )


if __name__ == "__main__":
    main()

"""
tests/test_pipeline.py
Unit tests for data pipeline and feature engineering.
Run: pytest tests/ -v --cov=src
"""
import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'pipeline'))

from data_pipeline import engineer_features, clean_data


# ── Fixtures ──────────────────────────────────────────────
@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'Time':   np.random.randint(0, 172800, n),
        'Amount': np.random.exponential(50, n),
        'Class':  np.random.choice([0, 1], n, p=[0.98, 0.02]),
    })
    for i in range(1, 29):
        df[f'V{i}'] = np.random.randn(n)
    return df


@pytest.fixture
def dirty_df(sample_df):
    df = sample_df.copy()
    df.loc[0, 'Amount'] = np.nan
    df = pd.concat([df, df.iloc[:5]])   # add duplicates
    return df


# ── Data Pipeline Tests ────────────────────────────────────
class TestDataPipeline:

    def test_engineer_features_adds_columns(self, sample_df):
        result = engineer_features(sample_df)
        expected = ['hour', 'is_night', 'log_amount', 'risk_score',
                    'v_mean', 'v_std', 'v1_v2', 'v14_v17']
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_log_amount_non_negative(self, sample_df):
        result = engineer_features(sample_df)
        assert (result['log_amount'] >= 0).all()

    def test_is_night_binary(self, sample_df):
        result = engineer_features(sample_df)
        assert set(result['is_night'].unique()).issubset({0, 1})

    def test_is_night_correct_hours(self, sample_df):
        df = sample_df.copy()
        df['Time'] = 3 * 3600   # 3am
        result = engineer_features(df)
        assert result['is_night'].iloc[0] == 1

    def test_clean_removes_nulls(self, dirty_df):
        result = clean_data(dirty_df)
        assert result.isnull().sum().sum() == 0

    def test_clean_removes_duplicates(self, dirty_df):
        before = len(dirty_df)
        result = clean_data(dirty_df)
        assert len(result) < before

    def test_risk_score_is_numeric(self, sample_df):
        result = engineer_features(sample_df)
        assert pd.api.types.is_float_dtype(result['risk_score'])

    def test_v_range_equals_max_minus_min(self, sample_df):
        result = engineer_features(sample_df)
        expected = result['v_max'] - result['v_min']
        pd.testing.assert_series_equal(result['v_range'], expected, check_names=False)


# ── Feature Value Range Tests ──────────────────────────────
class TestFeatureRanges:

    def test_hour_range(self, sample_df):
        result = engineer_features(sample_df)
        assert result['hour'].between(0, 23).all()

    def test_is_weekend_binary(self, sample_df):
        result = engineer_features(sample_df)
        assert set(result['is_weekend'].unique()).issubset({0, 1})

    def test_amount_features_match(self, sample_df):
        result = engineer_features(sample_df)
        # is_large_amt should be 1 when Amount > 500
        large = result[result['Amount'] > 500]
        if len(large) > 0:
            assert (large['is_large_amt'] == 1).all()

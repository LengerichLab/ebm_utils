import numpy as np
import pandas as pd
import pytest

from ebm_utils.analysis.changepoints import (
    find_and_plot_non_monotonicities,
    find_non_monotonicities,
    find_and_plot_discontinuities,
    find_discontinuities
)

# Sample data for testing
SAMPLE_X = pd.DataFrame({
    "feature1": np.random.rand(100),
    "feature2": np.random.rand(100)
})
SAMPLE_Y = np.random.randint(2, size=100)

@pytest.fixture
def sample_data():
    """Fixture to provide sample data."""
    return SAMPLE_X, SAMPLE_Y

# Test for find_non_monotonicities
def test_find_non_monotonicities(sample_data):
    X_train, Y_train = sample_data
    result = find_non_monotonicities(X_train, Y_train)
    assert isinstance(result, pd.DataFrame)

# Test for find_and_plot_non_monotonicities
def test_find_and_plot_non_monotonicities(sample_data):
    X_train, Y_train = sample_data
    result = find_and_plot_non_monotonicities(X_train, Y_train)
    assert isinstance(result, pd.DataFrame)

# Test for find_discontinuities
def test_find_discontinuities(sample_data):
    X_train, Y_train = sample_data
    result = find_discontinuities(X_train, Y_train)
    assert isinstance(result, pd.DataFrame)

# Test for find_and_plot_discontinuities
def test_find_and_plot_discontinuities(sample_data):
    X_train, Y_train = sample_data
    result = find_and_plot_discontinuities(X_train, Y_train)
    assert isinstance(result, pd.DataFrame)

# Add more detailed tests for each function
# These tests should check not only the types of the outputs
# but also the correctness of the data processing and results.

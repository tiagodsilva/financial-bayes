# AGENTS.md - Agent Coding Guidelines for financial-bayes

## Project Overview

This is a Jupyter notebook-based Python project for financial analysis. The primary code lives in `notebooks/` as `.ipynb` files.

## Environment Setup

### Prerequisites
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation
```bash
# Install dependencies using uv
uv sync

# Activate virtual environment
source .venv/bin/activate

# Add new dependencies
uv add <package>
uv add --dev <package>
```

## Build, Lint, and Test Commands

### Linting with Ruff
```bash
# Run ruff linter on all Python files
ruff check .

# Run ruff with auto-fix
ruff check --fix .

# Format code with ruff
ruff format .
```

### Running Notebooks
```bash
# Run a notebook (requires ipykernel)
jupyter notebook notebooks/<notebook-name>.ipynb

# Or use VS Code/Cursor with Jupyter extension
```

### No Traditional Tests
This project does not have traditional unit tests. Notebooks are tested manually through execution.

## Code Style Guidelines

### General Principles
- Keep code clean, readable, and well-documented
- Use meaningful variable and function names
- Prefer explicit over implicit
- Write code suitable for financial/scientific analysis

### Import Conventions
```python
# Standard library first
import os
import sys
from datetime import datetime
from dateutil import relativedelta

# Third-party libraries (alphabetical)
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import statsmodels.api as sm
import yfinance

# Local imports
from statsmodels.graphics.tsaplots import plot_acf
```

### Naming Conventions
- **Variables/functions**: `snake_case` (e.g., `stock_returns`, `calculate_volatility`)
- **Classes**: `PascalCase` (e.g., `DataLoader`, `Model`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_LOOKBACK_DAYS`)
- **Private variables**: `_leading_underscore` (e.g., `_cache`)

### Type Hints
Use type hints when beneficial for clarity:
```python
def calculate_returns(prices: np.ndarray, periods: int = 1) -> np.ndarray:
    """Calculate returns for given price series."""
    ...

def load_data(ticker: str, start_date: str) -> pd.DataFrame:
    ...
```

### Jupyter Notebook Best Practices
1. Start with markdown cell explaining the analysis objective
2. Include a TL;DR summary cell after the introduction
3. Add instruction cells for environment setup if needed
4. Use descriptive markdown headers (`##`, `###`) to organize sections
5. Keep code cells focused and reasonably sized
6. Include visualization cells with clear titles and labels
7. Add comments in code cells for complex logic

### Error Handling
- Use try/except blocks for external data fetching (e.g., yfinance)
- Provide informative error messages
- Handle missing data explicitly (don't silently propagate NaNs)

### Documentation
- Every notebook should have a clear title and objective
- Use markdown cells to explain methodology and conclusions
- Include citations for data sources
- Document statistical methods used

### Data Handling
- Prefer pandas DataFrames for tabular data
- Use numpy for numerical computations
- Handle datetime objects consistently
- Validate data after loading from external sources

### Visualization
- Always include titles and axis labels on plots
- Use appropriate figure sizes for readability
- Consider using seaborn for statistical visualizations
- Save figures in appropriate formats (PNG for quick views, PDF for publications)

## File Structure
```
finances/
├── .venv/              # Virtual environment
├── notebooks/          # Jupyter notebooks (main code)
│   └── *.ipynb
├── pyproject.toml     # Project configuration
├── uv.lock            # Locked dependencies
├── README.md          # Project overview
└── AGENTS.md          # This file
```

## Dependencies
Key libraries used in this project:
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `matplotlib` - Visualization
- `statsmodels` - Statistical modeling
- `scikit-learn` - Machine learning
- `torch` - Deep learning
- `yfinance` - Yahoo Finance data
- `ruff` - Linting and formatting
- `ipykernel` - Jupyter kernel

## Git Conventions
- Use descriptive commit messages
- Commit frequently with logical changes
- Keep notebooks in a runnable state

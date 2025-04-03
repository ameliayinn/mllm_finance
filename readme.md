## Usage

### 1. Prepare Your Data
Ensure your CSV file follows the structure:
- `body`: The text input for BERT.
- `past_50`: List of past 50 time steps of the time series.
- `future_10`: List of future 10 time steps (to be predicted).

### 2. Train the Model
Run the `training.py` script:

```bash
python training.py


# QR Decomposition on Embeddings

This project contains a series of experiments on decomposing embeddings using QR decomposition. The implementation explores the application of QR decomposition techniques on high-dimensional embedding spaces, particularly focusing on their transformation and analysis.

## Requirements

- Python 3.12+
- Poetry for dependency management

## Installation

1. Clone the repository:
```bash
git clone git@github.com:nishtahir/text-embeddings-decomposition.git
cd text-embeddings-decomposition
```

2. Setup Virtual Environment and Install dependencies using Poetry:
```bash
python -m venv .venv
source .venv/bin/activate

pip install poetry
poetry install
```

3. Set up environment variables:
Create a `.env` file in the root directory with necessary configurations.

```bash
OPENAI_API_KEY=your_openai_api_key
```

4. Prepare the dataset:

```bash
python qr_decomposition/main.py prepare-dataset
```

5. Train the classification model:

```bash
python qr_decomposition/main.py train-model
```


## Project Structure

```
qr_decomposition/
├── notebooks/          # Jupyter notebooks for experiments
├── qr_decomposition/   # Main package
│   ├── decomposition.py    # QR decomposition implementation
│   ├── distance.py         # Distance computation utilities
│   ├── embedding.py        # Embedding handling
│   ├── main.py            # Main application entry
│   └── sentiment/         # Sentiment analysis module
├── dataset/           # Data storage
├── test/             # Test suite
└── pyproject.toml    # Project configuration
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
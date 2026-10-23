# Dragon Data Processing

## Installation

Requires Python 3.12+.

```bash
git clone https://github.com/PolarCoasts/dragon_processing.git
cd dragon_processing
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

To use the environment in Jupyter, register it as a kernel:

```bash
pip install ipykernel
python -m ipykernel install --user --name dragon-processing --display-name "Python (dragon_processing)"
```

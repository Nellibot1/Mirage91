# Mirage91 Environment

This project works best in an isolated Python `venv` using Python `3.11`.

## Create The Environment

From this folder:

```bash
chmod +x create_env_mirage91.sh
./create_env_mirage91.sh
```

If your Python 3.11 binary has a different name or path:

```bash
PYTHON_BIN=/path/to/python3.11 ./create_env_mirage91.sh
```

## Activate

```bash
source mirage91/bin/activate
```

## Verify

```bash
python -c "import numpy, pandas, scipy, sklearn, joblib, matplotlib, pyxdf; print('ok')"
```

## Run

```bash
python ClassifierVersion4_2.py
python ClassifierVersion5.py
```

## Important

After activating the environment, use `python`, not `/opt/homebrew/bin/python3` or another system Python, otherwise the installed packages may not be found.

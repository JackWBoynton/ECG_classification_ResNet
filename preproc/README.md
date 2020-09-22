# Filtering and Segmentation

## Requirements
Data files can be retrieved from [Physionet](https://doi.org/10.13026/C2V88N).

Requires [WFDB](https://github.com/MIT-LCP/wfdb-python) for reading data files.

```
pip install wfdb
```

## Usage
The script can be used in a terminal.
```
usage: preproc.py [-h] file ecg_output ann_output

positional arguments:
  file        Record name without extension.
  ecg_output  ECG output filename
  ann_output  Annotation output filename

optional arguments:
  -h, --help  show this help message and exit
```

### Example
```
python preproc.py I01 ecg.npy ann.npy
```

# ECG Augmentation

ECG Augmentation is a Python library for augmenting 12-lead ECG signals.
<p align="center">
    <a href="https://github.com/badges/shields/graphs/contributors" alt="Contributors">
    <img src="https://img.shields.io/pypi/dw/ecgaugmentation" /></a>
</p>

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install ECG Augmentation.

```bash
pip install ecgaugmentation
```

## Usage

```python
import ecgaugmentation

in_path = "." # path to -fecg.npy ECG files
iterations = 100
anomaly_type = "RBBB"

ecgs, annotations = ecgaugmentation.augment(in_path, out_path, anomaly_type)
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)

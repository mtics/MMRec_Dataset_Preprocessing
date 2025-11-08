# Multimodal Recommendation Dataset Processing Framework

This is a unified framework for processing various multimodal recommendation datasets, solving the problems of fragmented code, inconsistent processing methods, and mismatched mapping indices.

## Key Features

- **Unified Interface**: Provides a unified processing interface for different types of datasets
- **Unified Mapping**: All datasets use consistent ID mapping (consecutive integers starting from 1)
- **Extensibility**: Supports adding new dataset types
- **High-Performance Image Download**: Multi-threaded image downloader with resume capability and intelligent retry
- **Configuration Management**: Centralized configuration management for easy maintenance and expansion
- **Modular Architecture**: Clear layered architecture that is easy to understand and maintain

## Project Structure

```
├── main.py                      # Main program entry
├── requirements.txt             # Project dependencies
└── src/                        # Source code directory
    ├── cli/                    # Command-line interface
    │   └── commands.py         # Command implementation
    ├── downloaders/            # Downloader modules
    │   └── image_downloader.py # Image downloader
    ├── managers/               # Manager modules
    │   ├── config_manager.py   # Configuration management
    │   └── dataset_manager.py  # Dataset management
    ├── models/                 # Data models
    │   └── dataset_config.py   # Dataset configuration model
    ├── processors/             # Data processors
    │   ├── base.py            # Base processor
    │   ├── amazon.py          # Amazon dataset processor
    │   └── movielens.py       # MovieLens dataset processor
    └── utils/                  # Utility modules
        └── logging_config.py   # Logging configuration
```

## Supported Datasets

### Amazon Datasets
- **Clothing** - Clothing, Shoes and Jewelry (5.7M ratings, 1.1M items)
- **Sports** - Sports and Outdoors
- **Baby** - Baby Products

### MovieLens Datasets
- **ML** - ml-latest-small (100K ratings, 9K movies)

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Basic Usage

```bash
# Show help information
python main.py --help

# List all available datasets
python main.py --list

# Process all datasets (including downloading images)
python main.py

# Process specific datasets
python main.py --datasets Baby Sports

# Only process data without downloading images
python main.py --skip-images

# Use more threads to accelerate image download (default 8 threads)
python main.py --datasets Clothing --max-workers 16

# View dataset configuration
python main.py --show-config Baby
```

## Output Format

Processed datasets will generate files in the following unified format under the `processed/` directory:

### ratings.csv
- `userID`: User ID (consecutive integers starting from 1)
- `itemID`: Item ID (consecutive integers starting from 1)
- `rating`: Rating value
- `timestamp`: Timestamp

### user_pairs.csv
- `originalUserID`: Original user ID
- `userID`: Mapped user ID

### item_pairs.csv
- `originalItemID`: Original item ID
- `itemID`: Mapped item ID
- `title`: Item title
- `img_url`: Item image URL


## Performance Optimization

### Image Download Optimization
- **Multi-threading**: Default 8 threads, adjustable via `--max-workers`
- **Intelligent Retry**: Different retry strategies for different error types
- **Resume Capability**: Automatically skips already downloaded images
- **Dynamic Delay**: Dynamically adjusts delay based on download results

### Large Dataset Processing
- **Vectorized Operations**: Uses pandas vectorized operations instead of loops, significantly improving performance
- **Chunked Saving**: Automatically saves in chunks when records exceed 1 million
- **Streaming Processing**: Reads large metadata files line by line to avoid memory overflow

## Extension Guide

### Adding New Datasets

1. Add configuration in `src/managers/config_manager.py`:

```python
# Amazon dataset
self.add_amazon_config(
    name="Books",
    ratings_file="amazon/dataset/ratings_Books.csv",
    metadata_file="amazon/dataset/meta_Books.json",
    output_dir="processed/Books"
)

# MovieLens dataset
self.add_movielens_config(
    name="ML-1M",
    ratings_file="ml-1m/ratings.dat",
    movies_file="ml-1m/movies.dat",
    output_dir="processed/ML-1M"
)
```

2. To customize processing logic, inherit the corresponding processor class and override methods.

## Citation

If you find this framework helpful in your research, please consider citing our paper:

```bibtex
@misc{li2025federatedvisionlanguagerecommendationpersonalizedfusion,
      title={Federated Vision-Language-Recommendation with Personalized Fusion},
      author={Zhiwei Li and Guodong Long and Jing Jiang and Chengqi Zhang and Qiang Yang},
      year={2025},
      eprint={2410.08478},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2410.08478},
}
```

## Contact

- This project is free for academic usage. You can run it at your own risk.
- For any other purposes, please contact Mr. Zhiwei Li ([lizhw.cs@outlook.com](mailto:lizhw.cs@outlook.com))


## Thanks

In the implementation of this project, we referred to the code of [MMRec](https://github.com/enoche/MMRec), and we are grateful for their open-source contributions!

## Contributing

Issues and Pull Requests are welcome to improve this framework.

## License

MIT License
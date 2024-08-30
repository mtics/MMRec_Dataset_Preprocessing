# MMRec_Dataset_Preprocessing

> This project is used to preprocess the multimodal recommendation datasets with foundation models.

Currently, we used the `open_clip` as the foundation model, and only preprocess the `title` field as the text modality.

## Requirements

1. The code is implemented with `Python ~= 3.8`;
2. Other requirements can be installed by `pip install -r requirements.txt`.

## Quick Start

1. Put datasets into the path `[parent_folder]/datasets/`;

2. For quick start, please run:
    ``````
    python main.py
    ``````

## Thanks

In the implementation of this project, we referred to the code of [MMRec](https://github.com/enoche/MMRec), and we are grateful for their open-source contributions!



## Contact

- This project is free for academic usage. You can run it at your own risk.
- For any other purposes, please contact Mr. Zhiwei Li ([lizhw.cs@outlook.com](mailto:lizhw.cs@outlook.com))
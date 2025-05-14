# -*- coding: utf-8 -*-
"""
Module Name: 
Description: 

Copyright (C) 2025 J.Cincotta

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
import os
import json
import numpy as np
import pandas as pd
from wildlife_tools.data import WildlifeDataset
from wildlife_datasets.datasets import utils
from typing import Callable
import pandas as pd


class WelfareObsDataset(WildlifeDataset):
    """
    PyTorch-style dataset for a datasets from wildlife-datasets library.

    Args:
        metadata_filename: A CSV file containing image metadata. Headings should contain:
            path
            identity
            segmentation
                (a list containing)
                counts
                size
            bbox
        root: Root directory if paths in metadata are relative. If None, absolute paths in metadata are used.
        transform: A function that takes in an image and returns its transformed version.
        img_load: Method to load images.
            Options: 'full', 'full_mask', 'full_hide', 'bbox', 'bbox_mask', 'bbox_hide',
                      and 'crop_black'.
        col_path: Column name in the metadata containing image file paths.
        col_label: Column name in the metadata containing class labels.
        load_label: If False, \_\_getitem\_\_ returns only image instead of (image, label) tuple.

    Attributes:
        labels np.array : An integers array of ordinal encoding of labels.
        labels_string np.array: A strings array of original labels.
        labels_map dict: A mapping between labels and their ordinal encoding.
        num_classes int: Return the number of unique classes in the dataset.
    """
    def __init__(
            self,
            root: str | None = None,
            annotations_file: str | None = None,
            transform: Callable | None = None,
            img_load: str = "full",
            col_path: str = "path",
            col_label: str = "identity",
            load_label: bool = True,
    ):
        # old pure CSV file version
        # metadata: pd.DataFrame = pd.read_csv(metadata_filename)
        # metadata = metadata.reset_index(drop=True)
        # based on datasets_wildme.py from wildlife-datasets
        path_json = os.path.join(root, 'annotations', annotations_file)
        path_images =  os.path.join(root, 'images')
        with open(path_json) as file:
            data = json.load(file)
        # Extract the data from the JSON file
        create_dict = lambda i: {
            'identity': i['category_id'],
            'bbox': i['bbox'],
            'image_id': i['image_id'],
            'category_id': i['category_id']
        }
        df_annotation = pd.DataFrame([create_dict(i) for i in data['annotations']])
        create_dict = lambda i: {
            'file_name': i['file_name'],
            'image_id': i['id']
        }
        df_images = pd.DataFrame([create_dict(i) for i in data['images']])
        species = pd.DataFrame(data['categories'])
        species = species.rename(columns={'id': 'category_id', 'name': 'species'})

        # Merge the information from the JSON file
        df = pd.merge(df_annotation, species, how='left', on='category_id')
        df = pd.merge(df, df_images, how='left', on='image_id')

        # Modify some columns
        df['path'] = path_images + os.path.sep + df['file_name']
        df['id'] = range(len(df))
        df.loc[df['identity'] == '____', 'identity'] = -1

        # Remove superficial columns
        df = df.drop(['image_id', 'file_name', 'category_id'], axis=1)
        # df = df.drop(['image_id', 'file_name', 'supercategory', 'category_id'], axis=1)
        df.rename({'id': 'image_id'}, axis=1, inplace=True)
        super().__init__(df, path_images, transform, img_load, col_path, col_label, load_label)
        
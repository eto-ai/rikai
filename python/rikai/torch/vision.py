#  Copyright 2021 Rikai Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""`Torchvision`_ compatible Dataset


.. _torchvision: https://pytorch.org/vision/stable/index.html
"""

from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import PIL

import rikai.torch.data
from rikai.torch.transforms import RikaiToTensor

__all__ = ["Dataset"]


class Dataset(rikai.torch.data.Dataset):
    """A Rikai Dataset compatible with `torchvision`_.

    Parameters
    ----------
    uri_or_df : str, Path, or pyspark.sql.DataFrame
        URI of the dataset or the dataset as a pyspark DataFrame
    image_column : str
        The column name for the image data.
    target_column : str or list[str]
        The column(s) of the target / label.
    transform : Callable, optional
        A function/transform that takes in an :py:class:`PIL.Image.Image` and
        returns a transformed version. E.g,
        :py:class:`torchvision.transforms.ToTensor`
    target_transform : Callable, optional
        A function/transform that takes in the target and transforms it.

    Yields
    ------
    (image, target)

    See Also
    --------
    `Torchvision Dataset <https://pytorch.org/vision/stable/datasets.html#>`_

    Examples
    --------
    >>> from torchvision import transforms
    >>> from rikai.torch.vision import Dataset
    >>> transform = transforms.Compose(
    ...     transforms=[
    ...         transforms.Resize(128),
    ...         transforms.ToTensor(),
    ...         transforms.Normalize(
    ...             (0.485, 0.456, 0.406),
    ...             (0.229, 0.224, 0.225)
    ...         ),
    ...     ])
    >>> dataset = Dataset("out", "image", ["label"], transform=transform)
    >>> next(iter(dataset))
    ... tensor([[[-1.8610, -0.8678, -0.4226,  ..., -1.7583,  0.0569, -0.6794],
         [-1.5870, -1.8782, -1.7069,  ..., -1.1075, -1.1760, -1.8782],
         [-2.1179, -0.5253, -1.7925,  ..., -0.3712, -1.4843, -1.2959],
         ...,
         [-1.1073, -0.3927, -0.8110,  ..., -0.9853,  0.1128, -1.0027]]]) dog

    """

    def __init__(
        self,
        uri_or_df: Union[str, Path, "pyspark.sql.DataFrame"],
        image_column: str,
        target_column: Union[str, List[str]],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.image_column = image_column
        self.target_columns = (
            [target_column]
            if isinstance(target_column, str)
            else target_column
        )
        super().__init__(
            uri_or_df,
            [self.image_column] + self.target_columns,
            transform=RikaiToTensor(use_pil=True),
        )

        self.transform = transform if transform else lambda x: x
        self.target_transform = (
            target_transform if target_transform else lambda x: x
        )

    def __repr__(self) -> str:
        return f"RikaiDataset({self.uri_or_df})"

    def __iter__(self) -> Tuple[PIL.Image.Image, Any]:
        for row in super().__iter__():
            image = row[self.image_column]
            target = tuple([row[col] for col in self.target_columns])
            if len(target) == 1:
                target = target[0]
            yield self.transform(image), self.target_transform(target)

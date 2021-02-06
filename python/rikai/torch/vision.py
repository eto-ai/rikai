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

"""`Torchvision`_ compatibile Dataset


.. _torchvision: https://pytorch.org/vision/stable/index.html
"""

from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import PIL

import rikai.torch.data

__all__ = ["Dataset"]


class Dataset(rikai.torch.data.Dataset):
    """A Rikai Dataset compatible with `torchvision`_.

    Parameters
    ----------
    uri : str or Path
        URI to the dataset
    image_column : str
        The column of the image.
    target_column : str or list[str]
        The column(s) of the target
    transform : Callable, optional
        A function/transform that takes in an :py:class:`PIL.Image.Image` and
        returns a transformed version. E.g, :py:class:`torchvision.transforms.ToTensor`
    target_transform : Callable, optional
        A function/transform that takes in the target and transforms it.

    Yields
    ------
    (image, target)

    See Also
    --------
    `Torchvision Dataset <https://pytorch.org/vision/stable/datasets.html#>`_

    """

    def __init__(
        self,
        uri: Union[str, Path],
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
        super().__init__(uri, [self.image_column] + self.target_columns)

        self.transform = transform if transform else lambda x: x
        self.target_transform = (
            target_transform if target_transform else lambda x: x
        )

    def __repr__(self) -> str:
        return f"RikaiVisionDataset({self.uri})"

    def __iter__(self) -> Tuple[PIL.Image.Image, Any]:
        for row in super().__iter__():
            image = row[self.image_column]
            target = tuple([row[col] for col in self.target_columns])
            if len(target) == 1:
                target = target[0]
            yield self.transform(image), self.target_transform(target)

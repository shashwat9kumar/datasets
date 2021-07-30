# coding=utf-8
# Copyright 2021 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""mslr_web dataset."""

import tensorflow_datasets.public_api as tfds
from tensorflow_datasets.ranking.mslr_web import mslr_web


class MslrWebTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for mslr_web dataset."""
  DATASET_CLASS = mslr_web.MslrWeb
  SPLITS = {
      "fold1_train": 6,
      "fold1_vali": 2,
      "fold1_test": 2,
      "fold2_train": 6,
      "fold2_vali": 2,
      "fold2_test": 2,
      "fold3_train": 6,
      "fold3_vali": 2,
      "fold3_test": 2,
      "fold4_train": 6,
      "fold4_vali": 2,
      "fold4_test": 2,
      "fold5_train": 6,
      "fold5_vali": 2,
      "fold5_test": 2
  }
  BUILDER_CONFIG_NAMES_TO_TEST = ["10k"]
  DL_EXTRACT_RESULT = {
      "10k": "10k",
      "30k": "30k",
  }
  OVERLAPPING_SPLITS = list(SPLITS.keys())


if __name__ == "__main__":
  tfds.testing.test_main()

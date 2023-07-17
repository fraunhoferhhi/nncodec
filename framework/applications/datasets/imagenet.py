'''
The copyright in this software is being made available under the Clear BSD
License, included below. No patent rights, trademark rights and/or 
other Intellectual Property Rights other than the copyrights concerning 
the Software are granted under this license.

The Clear BSD License

Copyright (c) 2019-2023, Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V. & The NNCodec Authors.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted (subject to the limitations in the disclaimer below) provided that
the following conditions are met:

     * Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.

     * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

     * Neither the name of the copyright holder nor the names of its
     contributors may be used to endorse or promote products derived from this
     software without specific prior written permission.

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
'''
import os

import pandas as pd
from torchvision import datasets, transforms

from framework.applications import settings

VALIDATION_FILES = os.path.join(settings.METADATA_DIR, 'imagenet_validation_files.txt')

transforms_pyt_model_zoo = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

class ImageNetDataset(datasets.ImageFolder):
    def __init__(self, root, *args, validate=False, train=True, use_precomputed_labels=False,
                 labels_path=None, **kwargs):
        """ImageNet root folder is expected to have two directories: train and val."""
        if train and validate == train:
            raise ValueError('Train and validate can not be True at the same time.')
        if use_precomputed_labels and labels_path is None:
            raise ValueError('If use_precomputed_labels=True the labels_path is necessary.')

        if train:
            root = os.path.join(root, 'train')
        elif validate:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'val')
        super().__init__(root, transform=transforms_pyt_model_zoo, *args, **kwargs)

        if validate and use_precomputed_labels:
            df = pd.read_csv(labels_path, sep='\t')
            df.input_path = df.input_path.apply(lambda x: os.path.join(root, x))
            mapping = dict(zip(df.input_path, df.pred_class))
            # self.samples = [(mapping[x[0]], x[1]) for x in self.samples]
            self.samples = [(x[0], mapping[x[0]]) for x in self.samples]
            self.targets = [x[1] for x in self.samples]

        if validate:
            with open(VALIDATION_FILES, 'r') as f:
                names = [x.strip() for x in f.readlines()]
            class_names = [x.split('_')[0] for x in names]
            val_names = set(os.path.join(self.root, class_name, x) for class_name, x in zip(class_names, names))
            self.samples = [x for x in self.samples if x[0] in val_names]
            self.targets = [x[1] for x in self.samples]


        if train:
            with open(VALIDATION_FILES, 'r') as f:
                names = [x.strip() for x in f.readlines()]
            class_names = [x.split('_')[0] for x in names]
            val_names = set(os.path.join(self.root, class_name, x) for class_name, x in zip(class_names, names))
            self.samples = [x for x in self.samples if x[0] not in val_names]
            self.targets = [x[1] for x in self.samples]


def imagenet_dataloaders(root, split='test'):

    if split == 'train':
        train_data = ImageNetDataset(root=root,
                                     train=True,
                                     validate=False
                                     )
        return train_data

    elif split == 'val':
        val_data = ImageNetDataset(root=root,
                                   train=False,
                                   validate=True
                                   )
        return val_data

    elif split == 'test':
        test_data = ImageNetDataset(root=root,
                                    train=False,
                                    validate=False
                                    )
        return test_data

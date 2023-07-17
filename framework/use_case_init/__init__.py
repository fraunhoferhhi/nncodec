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
import numpy as np
import torch
import tensorflow as tf
from torch.utils.data import dataloader, random_split
from framework.applications.datasets import imagenet
from framework.applications.utils import evaluation, train, transforms

class ModelSetting:

    def __init__(self, model_transform, evaluate, train, dataset, criterion):

        self.model_transform = model_transform
        self.evaluate = evaluate
        self.train = train
        self.dataset = dataset
        self.criterion = criterion
        self.train_cf = False

    def init_training(self, dataset_path, batch_size, num_workers):
        train_set = self.dataset(
            root=dataset_path,
            split='train'
        )
        train_loader = torch.utils.data.DataLoader(
                    train_set,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=True,
                    collate_fn=getattr(train_set, "collate_fn", dataloader.default_collate),
                    sampler=getattr(train_set, "sampler", None),
                )

        return train_loader

    def init_test(self, dataset_path, batch_size, num_workers):
        test_set = self.dataset(
            root=dataset_path,
            split='test'
        )

        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=getattr(test_set, "collate_fn", dataloader.default_collate),
            sampler=getattr(test_set, "sampler", None),
        )
        return test_set, test_loader

    
    def init_validation(self, dataset_path, batch_size, num_workers):
        val_set = self.dataset(
            root=dataset_path,
            split='val'
        )

        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=getattr(val_set, "collate_fn", dataloader.default_collate),
            sampler=getattr(val_set, "sampler", None),
        )

        return val_set, val_loader

    def init_test_tef(self, dataset_path, batch_size, num_workers, model_name):
        test_set = self.dataset(
            root=dataset_path,
            split='test'
        )
        self.__model_name = model_name

        test_images, test_labels = zip(*test_set.imgs)
        test_loader = tf.data.Dataset.from_tensor_slices((list(test_images), list(test_labels)))
        test_loader = test_loader.map(lambda image, label: tf.py_function(self.preprocess,
                                                                              inp=[image, label],
                                                                              Tout=[tf.float32, tf.int32]), num_parallel_calls=num_workers).batch(
                                                                                batch_size)

        return test_set, test_loader

    def init_validation_tef(self, dataset_path, batch_size, num_workers, model_name):
        val_set = self.dataset(
            root=dataset_path,
            split='val'
        )
        self.__model_name = model_name

        val_images, val_labels = zip(*val_set.imgs)
        val_loader = tf.data.Dataset.from_tensor_slices((list(val_images), list(val_labels)))
        val_loader = val_loader.map(lambda image, label: tf.py_function(self.preprocess,
                                                                              inp=[image, label],
                                                                              Tout=[tf.float32, tf.int32]), num_parallel_calls=num_workers).batch(
                                                                                batch_size)

        return val_set, val_loader

    
    def preprocess(
                    self,
                    image,
                    label
    ):
        image_size = 224

        if self.__model_name == 'EfficientNetB1':
            image_size = 240
        elif self.__model_name == 'EfficientNetB2':
            image_size = 260
        elif self.__model_name == 'EfficientNetB3':
            image_size = 300
        elif self.__model_name == 'EfficientNetB4':
            image_size = 380
        elif self.__model_name == 'EfficientNetB5':
            image_size = 456
        elif self.__model_name == 'EfficientNetB6':
            image_size = 528
        elif self.__model_name == 'EfficientNetB7':
            image_size = 600

        image, label = self.model_transform(image, label, image_size=image_size)

        if 'DenseNet' in self.__model_name:
            return tf.keras.applications.densenet.preprocess_input(image), label
        elif 'EfficientNet' in self.__model_name:
            return tf.keras.applications.efficientnet.preprocess_input(image), label
        elif self.__model_name == 'InceptionResNetV2':
            return tf.keras.applications.inception_resnet_v2.preprocess_input(image), label
        elif self.__model_name == 'InceptionV3':
            return tf.keras.applications.inception_v3.preprocess_input(image), label
        elif self.__model_name == 'MobileNet':
            return tf.keras.applications.mobilenet.preprocess_input(image), label
        elif self.__model_name == 'MobileNetV2':
            return tf.keras.applications.mobilenet_v2.preprocess_input(image), label
        elif 'NASNet' in self.__model_name:
            return tf.keras.applications.nasnet.preprocess_input(image), label
        elif 'ResNet' in self.__model_name and 'V2' not in self.__model_name:
            return tf.keras.applications.resnet.preprocess_input(image), label
        elif 'ResNet' in self.__model_name and 'V2' in self.__model_name:
            return tf.keras.applications.resnet_v2.preprocess_input(image), label
        elif self.__model_name == 'VGG16':
            return tf.keras.applications.vgg16.preprocess_input(image), label
        elif self.__model_name == 'VGG19':
            return tf.keras.applications.vgg19.preprocess_input(image), label
        elif self.__model_name == 'Xception':
            return tf.keras.applications.xception.preprocess_input(image), label


# supported use cases
use_cases = {
    "NNR_PYT":  ModelSetting( None,
                          evaluation.evaluate_classification_model,
                          train.train_classification_model,
                          imagenet.imagenet_dataloaders,
                          torch.nn.CrossEntropyLoss()
                          ),

    "NNR_TEF": ModelSetting( transforms.transforms_tef_model_zoo,
                            evaluation.evaluate_classification_model_TEF,
                            None,
                            imagenet.imagenet_dataloaders,
                            torch.nn.CrossEntropyLoss
                            )
}
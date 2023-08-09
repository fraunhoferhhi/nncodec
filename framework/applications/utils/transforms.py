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

"""
Defines data and model transforms.
"""

import torch
import copy
import torch.nn as nn
from torch.functional import F
import cv2 as cv
import numpy as np


def transforms_tef_model_zoo(filename, label, image_size=224):

    img = cv.imread(filename.numpy().decode()).astype(np.float32)

    resize = 256
    if image_size > 224:
        resize = image_size

    # Resize
    height, width, _ = img.shape
    new_height = height * resize // min(img.shape[:2])
    new_width = width * resize // min(img.shape[:2])
    img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_CUBIC)

    # Crop
    height, width, _ = img.shape
    startx = width // 2 - (image_size // 2)
    starty = height // 2 - (image_size // 2)
    img = img[starty:starty + image_size, startx:startx + image_size]
    assert img.shape[0] == image_size and img.shape[1] == image_size, (img.shape, height, width)

    # BGR to RGB
    img = img[:, :, ::-1]

    return img, label

class ScaledConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.weight_scaling = nn.Parameter(torch.ones_like(torch.Tensor(out_channels, 1, 1, 1)))
        # self.reset_parameters()

    def reset_parameters(self):
        # The if condition is added so that the super call in init does not reset_parameters as well.
        if hasattr(self, 'weight_scaling'):
            nn.init.normal_(self.weight_scaling, 1, 1e-5)
            super().reset_parameters()

    def forward(self, input):
        torch_version_str = str(torch.__version__).split('.')
        if int(torch_version_str[0]) >= 1 and int(torch_version_str[1]) > 7:
            return self._conv_forward(input, self.weight_scaling * self.weight, self.bias)
        else:
            return self._conv_forward(input, self.weight_scaling * self.weight)

class ScaledLinear(nn.Linear):
    def __init__(self, in_features, out_features, *args, **kwargs):
        super().__init__(in_features, out_features, *args, **kwargs)
        self.weight_scaling = nn.Parameter(torch.ones_like(torch.Tensor(out_features, 1)))
        # self.reset_parameters()

    def reset_parameters(self):
        # The if condition is added so that the super call in init does not reset_parameters as well.
            if hasattr(self, 'weight_scaling'):
                nn.init.normal_(self.weight_scaling, 1, 1e-5)
                super().reset_parameters()

    def forward(self, input):
        return F.linear(input, self.weight_scaling * self.weight, self.bias)

class LSA:
    def __init__(self,
                 original_model):

        self.mdl = copy.deepcopy(original_model)

    def update_conv2d(self, m, parent):
        lsa_update = ScaledConv2d(m[1].in_channels, m[1].out_channels, m[1].kernel_size, m[1].stride,
                                  m[1].padding, m[1].dilation, m[1].groups, None, m[1].padding_mode)
        lsa_update.weight, lsa_update.bias = m[1].weight, m[1].bias
        setattr(parent, m[0], lsa_update)

    def update_linear(self, m, parent):
        lsa_update = ScaledLinear(m[1].in_features, m[1].out_features)
        lsa_update.weight, lsa_update.bias = m[1].weight, m[1].bias
        setattr(parent, m[0], lsa_update)

    def add_lsa_params(self):
        '''
        adds LSA scaling parameters to conv and linear layers
            - max. nested object depth: 4
            - trainable_true (i.e. does not add LSA params to layers which are not trained, e.g. in classifier only training)
        '''
        for m in self.mdl.named_children():
            if isinstance(m[1], nn.Conv2d) and m[1].weight.requires_grad:
                self.update_conv2d(m, self.mdl)
            elif isinstance(m[1], nn.Linear) and m[1].weight.requires_grad:
                self.update_linear(m, self.mdl)
            elif len(dict(m[1].named_children())) > 0:
                for n in m[1].named_children():
                    if isinstance(n[1], nn.Conv2d) and n[1].weight.requires_grad:
                        self.update_conv2d(n, m[1])
                    elif isinstance(n[1], nn.Linear) and n[1].weight.requires_grad:
                        self.update_linear(n, m[1])
                    elif len(dict(n[1].named_children())) > 0:
                        for o in n[1].named_children():
                            if isinstance(o[1], nn.Conv2d) and o[1].weight.requires_grad:
                                self.update_conv2d(o, n[1])
                            elif isinstance(o[1], nn.Linear) and o[1].weight.requires_grad:
                                self.update_linear(o, n[1])
                            elif len(dict(o[1].named_children())) > 0:
                                for p in o[1].named_children():
                                    if isinstance(p[1], nn.Conv2d) and p[1].weight.requires_grad:
                                        self.update_conv2d(p, o[1])
                                    elif isinstance(p[1], nn.Linear) and p[1].weight.requires_grad:
                                        self.update_linear(p, o[1])
                                    elif len(dict(p[1].named_children())) > 0:
                                        for q in p[1].named_children():
                                            if isinstance(q[1], nn.Conv2d) and q[1].weight.requires_grad:
                                                self.update_conv2d(q, p[1])
                                            elif isinstance(q[1], nn.Linear) and q[1].weight.requires_grad:
                                                self.update_linear(q, p[1])
        return self.mdl
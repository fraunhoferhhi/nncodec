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
import logging
import torch
import numpy as np
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

def freeze_batch_norm_layers(model):
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.eval()

def train_classification_model(model, optimizer, criterion, trainloader, device,
                         verbose=True, max_batches=None, freeze_batch_norm=False):
    """
    Parameters
    ----------
    device: torch.device
        Choose between cuda or cpu.
    model: torch.nn.Module
        A pytorch network model.
    optimizer: torch.optim.Optimizer
        A pytorch optimizer like Adam.
    criterion: torch.nn.Loss
        A pytorch criterion that defines the loss.
    trainloader: torch.utils.data.DataLoader
        Loader of train data.
    max_batches: int
        How many batches the model should train for.
    verbose: bool
        If True, print text - verbose mode.
    freeze_batch_norm: bool
        If True set batch norm layers to eval. Default: False

    Returns
    -------
    success: bool
        Returns False is nans encountered in the loss else True.
    """
    model.to(device)
    model.train()
    if freeze_batch_norm:
        freeze_batch_norm_layers(model)

    train_loss = []
    correct = 0
    total = 0

    total_iterations = max_batches or len(trainloader)
    iterator = tqdm(enumerate(trainloader), total=total_iterations, position=0, leave=True, desc='train_classification') \
        if verbose else enumerate(trainloader)

    for batch_idx, (inputs, targets) in iterator:

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)

        if torch.isnan(loss):
            LOGGER.warning('--> Loss is Nan.')
            break

        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

        _, predicted = outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx == max_batches:
            break

    acc = correct * 100.0 / total
    mean_train_loss = np.mean(train_loss)
    
    return acc, mean_train_loss

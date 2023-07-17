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

import copy
import torch
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from tqdm import tqdm

from framework.applications.utils.metrics import get_topk_accuracy_per_batch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_classification_model(model, criterion, testloader, testset,  min_sample_size=1000, max_batches=None,
                                  early_stopping_threshold=None, device=DEVICE, print_classification_report=False,
                                  return_predictions=False, verbose=False):
    """
    Helper function to evaluate model on test dataset.

    Parameters
    ----------
    model: torch.nn.Module
        Neural network model.
    criterion: torch.nn.Criterion
        Criterion for loss calculation.
    testloader: torch.utils.data.DataLoader
        DataLoader that loaded testset.
    testset: torch.utils.data.dataset.Dataset
        Test dataset
    min_sample_size: int
        Minimum sample size used for early_stopping calculation. Default: 1000
    max_batches: int
        Maximum batches evaluated, by default evaluates the complete testset. Default: None
    early_stopping_threshold: int
        A value between 0-100 corresponding to the accuracy. If it drops under a given threshold
        the evaluation is stopped.
    device: str
        Device on which the model is evaluated: cpu or cuda.
    print_classification_report: bool
        If True print the complete confusion matrix for all the classes.
    return_predictions: bool
        If True return all the predictions for all samples, otherwise return the accuracy.
    verbose: bool
        If True print the progress bar of the evaluation.

    Return
    ------
    output: float | nd.array
        Accuracy or all predictions, depending on the given return_predictions parameter.
    """
    model = model.to(device)
    model.eval()
    test_loss = []
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0
    top5_acc = 0

    # set (verbose) iterator
    total_iterations = max_batches or len(testloader)
    iterator = tqdm(enumerate(testloader), total=total_iterations, position=0, leave=True) if verbose else enumerate(testloader)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in iterator:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, targets)

            if outputs.size(1) > 5:
                c1, c5 = get_topk_accuracy_per_batch(outputs, targets, topk=(1, 5))
                top5_acc += c5 * targets.size(0)

            test_loss.append(loss.item())
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            all_predictions.append(np.array(predicted.cpu()))
            all_labels.append(np.array(targets.cpu()))

            acc = 100. * correct / total
            if batch_idx == max_batches:
                break
            elif len(all_predictions) > min_sample_size and early_stopping_threshold is not None and \
                    acc < early_stopping_threshold:
                break

        acc = 100. * correct / total
        if top5_acc != 0:
            top5_acc = top5_acc / total

        if print_classification_report:
            print(classification_report(np.concatenate(all_labels), np.concatenate(all_predictions),
                                        target_names=list(testset.mapping.keys()),
                                        labels=list(testset.mapping.values())))

        if return_predictions:
            return np.concatenate(all_predictions)
        else:
            mean_test_loss = np.mean(test_loss)
            return acc, float(top5_acc), mean_test_loss


def evaluate_classification_model_TEF(model, test_loader, test_set, num_workers=8, verbose=0):

    _ , val_labels = zip(*test_set.imgs)

    y_pred = model.predict(test_loader, verbose=verbose, callbacks=None, max_queue_size=10, workers=num_workers,
                           use_multiprocessing=True)

    top5 = tf.keras.metrics.sparse_top_k_categorical_accuracy(val_labels, y_pred, k=5)
    top1 = tf.keras.metrics.sparse_categorical_accuracy(val_labels, y_pred)
    loss = tf.keras.metrics.sparse_categorical_crossentropy(val_labels, y_pred)

    acc = []
    acc.append((tf.keras.backend.sum(top1) / len(top1)).numpy() * 100)
    acc.append((tf.keras.backend.sum(top5) / len(top5)).numpy() * 100)
    acc.append((tf.keras.backend.mean(loss)).numpy())

    return acc

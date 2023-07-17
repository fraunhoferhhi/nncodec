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
import sys
import copy
import multiprocessing
import time
import traceback
from .. import common
import deepCABAC
from nnc_core.coder import hls
import nnc_core
from nnc_core.nnr_model import NNRModelAccess


def derive_sorted_codebook_from_tensor(tensor):
    originalShape = tensor.shape
    codebook, indices = np.unique(tensor, return_inverse=True)
    reshaped_indices = indices.reshape( originalShape )
    return codebook, reshaped_indices.astype('int32')

def get_codebook_offset( codebook, indices, cabac_unary_length_minus1 ):
    codebookOffset = 0
    if indices.dtype == np.int32:
        codebookOffset = -1
        minBits = None
        for cb in range( len( codebook ) ):
            encoder = deepCABAC.Encoder()
            encoder.initCtxModels( cabac_unary_length_minus1, 1 )
            indexes = indices - cb
            encoder.encodeLayer( indexes, 0, 0 )
            bits = len( encoder.finish().tobytes() )
            if minBits == None or bits < minBits:
                minBits = bits
                codebookOffset = cb

    indexes = indices - codebookOffset

    return codebook, indexes, codebookOffset

def get_best_egk(codebook, codebookOffset):
    cb_hls = {}
    cb_hls["CbZeroOffset__"] = codebookOffset
    cb_hls["codebook_size__"] = len( codebook )
    cb_hls["codebook__"] = codebook

    min_bytes_cb = None
    for i in range(16):
        cb_hls["codebook_egk__"] = i
        bs = bytearray()
        w = hls.BitWriter(bs)
        hls_enc = hls.Coder( w, cb_hls )
        hls_enc.codebook("")
        bytes_cb = w.getNumBitsTouched()
        if min_bytes_cb is None or bytes_cb < min_bytes_cb:
            min_bytes_cb = bytes_cb
            best_egk = i
    return best_egk, min_bytes_cb

def get_codebook_bytes(codebook, codebookOffset, cbEgk):
    cb_hls = {}
    cb_hls["CbZeroOffset__"] = codebookOffset
    cb_hls["codebook_size__"] = len( codebook )
    cb_hls["codebook__"] = codebook
    cb_hls["codebook_egk__"] = cbEgk
    
    bs = bytearray()
    w = hls.BitWriter(bs)
    hls_enc = hls.Coder( w, cb_hls )
    hls_enc.codebook("")
    bytes_cb = w.getNumBitsTouched()
   
    return bytes_cb

def approx(approx_info, model_info, approx_data_in, param_opt=0):

    ##Qunatize tensor with uniform but without DQ
    approx_data_out = {k: copy.copy(v) for k, v in approx_data_in.items()} # create copies of dicts in approx_data
    encoder = deepCABAC.Encoder()
    model_access = NNRModelAccess(model_info)
    for block_or_param in model_access.blocks_and_params():
        for par_type, param, _ in block_or_param.param_generator(approx_data_in["compressed_parameter_types"]):
            if (par_type in approx_info["to_approximate"]) and (param not in approx_data_in["approx_method"]):
                # !!! There seems to be a pybind11 issue when using np.zeros_like for "values" that have been transposed.
                # !!! It seems that sometimes, encoder.quantLayer returns only zeros for quantizedValues. Needs further study.
                # !!! For now, using np.zeros instead of np.zeros_like seems to be a workaround.
                quantizedValues = np.zeros(approx_data_in["parameters"][param].shape, dtype=np.int32)
                encoder.initCtxModels( approx_info["cabac_unary_length_minus1"], 0 )
                
                qp_off = 0
                if approx_info['dq_flag'][param] == 1:
                    qp_off = common.compute_qp_offset_to_dq_equivalent( approx_data_out['qp_density'] )
                    print("INFO: Dependent quatization (DQ) can not be used with 'codebook'. In order to get similiar performance (to DQ) the QP is changed by {}!".format(-qp_off))

                enc_qp = approx_info['qp'][param] - qp_off

                qp = encoder.quantLayer(
                    approx_data_in["parameters"][param],
                    quantizedValues,
                    0, #approx_info['dq_flag'][param],
                    approx_data_out['qp_density'],
                    enc_qp,
                    approx_info["lambda_scale"],
                    approx_info["cabac_unary_length_minus1"],
                    approx_data_in["scan_order"].get(param, 0)
                )
                if qp != enc_qp:
                    print("INFO: QP for {} has been clipped from {} to {} to avoid int32_t overflow!".format(param, approx_info['qp'][param],qp))
                    approx_data_out['qp'][param] = qp
                else:
                    approx_data_out['qp'][param] = enc_qp

                codebook, indexes = derive_sorted_codebook_from_tensor(quantizedValues)
                codebook, indexes, codebookOffset = get_codebook_offset( codebook, indexes,  approx_info["cabac_unary_length_minus1"])
                egk, _ = get_best_egk(codebook, codebookOffset)

                if approx_info["codebook_mode"] == 1:
                    approx_data_out["parameters"][param] = indexes
                    approx_data_out["codebooks"][param] = codebook
                    approx_data_out['approx_method'][param] = 'codebook'
                    approx_data_out['dq_flag'][param] = 0
                    approx_data_out["codebook_zero_offsets"][param] = codebookOffset
                    approx_data_out['codebooks_egk'][param] = egk
                
                elif approx_info["codebook_mode"] == 2:
                    if approx_info['dq_flag'][param] == 1:
                        quantizedValues = np.zeros(approx_data_in["parameters"][param].shape, dtype=np.int32)
                        encoder.initCtxModels( approx_info["cabac_unary_length_minus1"], 0 )
                        
                        enc_qp = approx_info['qp'][param]
                        ##else quantize again with DQ
                        qp = encoder.quantLayer(
                            approx_data_in["parameters"][param],
                            quantizedValues,
                            approx_info['dq_flag'][param],
                            approx_data_out['qp_density'],
                            enc_qp,
                            approx_info["lambda_scale"],
                            approx_info["cabac_unary_length_minus1"],
                            approx_data_in["scan_order"].get(param, 0)
                        )
                    

                    ##Compute Cost for encoding uniform quantized parameters
                    testEnc = deepCABAC.Encoder()
                    testEnc.initCtxModels( approx_info["cabac_unary_length_minus1"], param_opt )
                    testEnc.encodeLayer(quantizedValues, approx_info['dq_flag'][param], approx_data_in["scan_order"].get(param, 0) )
                    bs_par = bytearray( testEnc.finish().tobytes() )

                    bytesUni = len(bs_par)
                    ##Compute cost for codebook quantized parameters + bytes for encoding the codebooks
                    testEnc = deepCABAC.Encoder()
                    testEnc.initCtxModels( approx_info["cabac_unary_length_minus1"], param_opt )
                    testEnc.encodeLayer(indexes, 0, approx_data_in["scan_order"].get(param, 0) )
                    bs_par_cb = bytearray( testEnc.finish().tobytes() )

                    bytesCb = len(bs_par_cb) + get_codebook_bytes(codebook, codebookOffset, egk)

                    ##select cheapest
                    if bytesCb < bytesUni:
                        approx_data_out["parameters"][param] = indexes
                        approx_data_out["codebooks"][param] = codebook
                        approx_data_out['approx_method'][param] = 'codebook'
                        approx_data_out['dq_flag'][param] = 0
                        approx_data_out["codebook_zero_offsets"][param] = codebookOffset
                        approx_data_out['codebooks_egk'][param] = egk
                    else:
                        if approx_info['dq_flag'][param] == 1:
                            if qp != enc_qp:
                                print("INFO: QP for {} has been clipped from {} to {} to avoid int32_t overflow!".format(param, approx_info['qp'][param],qp))
                                approx_data_out['qp'][param] = qp
                            else:
                                approx_data_out['qp'][param] = enc_qp
                        approx_data_out['parameters'][param] = quantizedValues
                        approx_data_out['approx_method'][param] = 'uniform'
                        approx_data_out['dq_flag'][param] = approx_info['dq_flag'][param]

    approx_info_out = approx_info

    return approx_data_out, approx_info_out

# **********************************************************************************************************************
def rec(param, approx_data):
    assert approx_data['parameters'][param].dtype == np.int32
    cb = approx_data['codebooks'][param]
    stepsize = common.get_stepsize_from_qp(approx_data["qp"][param], approx_data["qp_density"])
    cb = cb * stepsize
    offset = approx_data['codebook_zero_offsets'][param]
    approx_data["parameters"][param] = cb[approx_data["parameters"][param] + offset]
    del approx_data["approx_method"][param]
    del approx_data['codebooks'][param]
    del approx_data['codebook_zero_offsets'][param]
    del approx_data['codebooks_egk'][param]
    del approx_data['qp'][param]
    

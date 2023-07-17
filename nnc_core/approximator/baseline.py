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
import numpy as np
import deepCABAC
import nnc_core
from nnc_core.nnr_model import NNRModelAccess
from nnc_core.coder import hls, baseline
from .. import common


def approx(approx_info, model_info, approx_data_in):
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

                enc_qp = approx_info['qp'][param]

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

                if qp != enc_qp:
                    print("INFO: QP for {} has been clipped from {} to {} to avoid int32_t overflow!".format(param, approx_info['qp'][param],qp))
                    approx_data_out['qp'][param] = qp
                else:
                    approx_data_out['qp'][param] = enc_qp

                approx_data_out['parameters'][param] = quantizedValues
                approx_data_out['approx_method'][param] = 'uniform'
                approx_data_out['dq_flag'][param] = approx_info['dq_flag'][param]
   
    return approx_data_out

def rec(param, approx_data):
    assert approx_data['parameters'][param].dtype == np.int32

    decoder = deepCABAC.Decoder()
    values = approx_data['parameters'][param]

    approx_data["parameters"][param] = np.zeros(values.shape, dtype=np.float32)
    decoder.dequantLayer(approx_data["parameters"][param], values, approx_data["qp_density"], approx_data["qp"][param], approx_data['scan_order'].get(param, 0))

    del approx_data["approx_method"][param]


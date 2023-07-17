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

from nnc_core import hls
import numpy as np


def encode(encoder, approx_data, param, ndu, mps, lps, param_opt_flag):
    if (
        (ndu["nnr_compressed_data_unit_payload_type"] == hls.CompressedDataUnitPayloadType.NNR_PT_FLOAT) or
        (ndu["nnr_compressed_data_unit_payload_type"] == hls.CompressedDataUnitPayloadType.NNR_PT_BLOCK)
    ):
        quantization_parameter =  lps["lps_quantization_parameter"] if lps is not None else mps["mps_quantization_parameter"]
        qp_density             =  lps["lps_qp_density"] if lps is not None else mps["mps_qp_density"]
        encoder.iae_v( 6 + qp_density, approx_data["qp"][param] - quantization_parameter) 
    
    encoder.initCtxModels( ndu["cabac_unary_length_minus1"], param_opt_flag )
    if param in approx_data["scan_order"]:
        assert ndu["scan_order"] == approx_data["scan_order"][param], "All parameters of a block must use the same scan_order."
    scan_order = ndu.get("scan_order", 0)
    if approx_data["parameters"][param].ndim <= 1:
        scan_order = 0
    encoder.encodeLayer(approx_data["parameters"][param], approx_data["dq_flag"][param], scan_order)


def decode( decoder, approx_data, param, ndu, mps, lps ):
    if (
        (ndu["nnr_compressed_data_unit_payload_type"] == hls.CompressedDataUnitPayloadType.NNR_PT_FLOAT) or
        (ndu["nnr_compressed_data_unit_payload_type"] == hls.CompressedDataUnitPayloadType.NNR_PT_BLOCK)
    ):
        quantization_parameter        =  lps["lps_quantization_parameter"] if lps is not None else mps["mps_quantization_parameter"]
        qp_density                    =  lps["lps_qp_density"] if lps is not None else mps["mps_qp_density"]
        approx_data["qp"][param]      = np.int32(decoder.iae_v( 6 + qp_density ) + quantization_parameter)
        approx_data["dq_flag"][param] = ndu["dq_flag"]
    
    else:
        approx_data["dq_flag"][param] = 0
        
    decoder.initCtxModels( ndu["cabac_unary_length_minus1"] )
    scan_order = ndu.get("scan_order", 0)
    if approx_data["parameters"][param].ndim <= 1:
        scan_order = 0
    decoder.decodeLayer(approx_data["parameters"][param], approx_data["dq_flag"][param], scan_order)


def decodeAndCreateEPs( decoder, approx_data, param, ndu, mps, lps ):
    if (
        (ndu["nnr_compressed_data_unit_payload_type"] == hls.CompressedDataUnitPayloadType.NNR_PT_FLOAT) or
        (ndu["nnr_compressed_data_unit_payload_type"] == hls.CompressedDataUnitPayloadType.NNR_PT_BLOCK)
    ):
        quantization_parameter        = lps["lps_quantization_parameter"] if lps is not None else mps["mps_quantization_parameter"]
        qp_density                    = lps["lps_qp_density"] if lps is not None else mps["mps_qp_density"]
        approx_data["qp"][param]      = np.int32(decoder.iae_v( 6 + qp_density ) + quantization_parameter)
        approx_data["dq_flag"][param] = ndu["dq_flag"]
        
    decoder.initCtxModels( ndu["cabac_unary_length_minus1"] )
    scan_order = ndu.get("scan_order", 0)
    if approx_data["parameters"][param].ndim <= 1:
        scan_order = 0
    entryPointArray = decoder.decodeLayerAndCreateEPs(approx_data["parameters"][param], approx_data.get("dq_flag", {}).get(param, 0), scan_order)

    return entryPointArray



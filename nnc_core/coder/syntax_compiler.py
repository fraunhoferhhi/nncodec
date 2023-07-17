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
from nnc_core import nnr_model

def compile_start_unit(profile):
    ndu_start = {}
    ndu_start["nnr_unit_type"] = hls.NnrUnitType.NNR_STR
    ndu_start["partial_data_counter_present_flag"] = 0
    ndu_start["partial_data_counter"] = 0
    ndu_start["independently_decodable_flag"] = 1

    # nnr_start_unit_header syntax elements
    ndu_start["general_profile_idc"] = profile

    return ndu_start

def compile_mps(approx_data, topology_present ):
    mps = {}
    # nnr_unit_header syntax elements
    mps["nnr_unit_type"] = hls.NnrUnitType.NNR_MPS
    mps["partial_data_counter_present_flag"] = 0
    mps["partial_data_counter"] = 0
    mps["independently_decodable_flag"] = 1
    
    # model_parameter_set_payload syntax elements
    mps["topology_carriage_flag"] = topology_present
    mps["mps_sparsification_flag"] = 0
    mps["mps_pruning_flag"] = 0
    mps["mps_unification_flag"] = 0
    mps["mps_decomposition_performance_map_flag"] = 0
    if "qp_density" in approx_data:
        mps["mps_quantization_method_flags"] = hls.QuantizationMethodFlags.NNR_QSU
        mps["mps_qp_density"] = approx_data["qp_density"]
        mps["mps_quantization_parameter"] = 0
    else:
        mps["mps_quantization_method_flags"] = 0
    
    mps["mps_topology_indexed_reference_flag"] = 0
    mps["nnr_reserved_zero_7bits"] = 0 
    
    return mps        


def compile_ndu_oob(tensor_dims = None, cabac_unary_length_minus1 = None, compressed_parameter_types = None, decomposition_parameter_dict = None ):
   ndu_oob = {}
   ndu_oob["input_parameters_present_flag"] = 1 if not all( [tensor_dims, cabac_unary_length_minus1, compressed_parameter_types, decomposition_parameter_dict and compressed_parameter_types & hls.BlockParameterTypes.NNR_CPT_DC != 0] ) else 0
   if tensor_dims is not None:
       ndu_oob["tensor_dimensions_flag"] = 0
       ndu_oob["tensor_dimensions"] = tensor_dims
       ndu_oob["count_tensor_dimensions"] = len(tensor_dims)
   else:
       ndu_oob["tensor_dimensions_flag"] = 1
   if cabac_unary_length_minus1 is not None:
       ndu_oob["cabac_unary_length_flag"] = 0
       ndu_oob["cabac_unary_length_minus1"] = cabac_unary_length_minus1
   else:
       ndu_oob["cabac_unary_length_flag"] = 1
   if compressed_parameter_types is not None:
       ndu_oob["compressed_parameter_types"] = compressed_parameter_types
   if decomposition_parameter_dict is not None and compressed_parameter_types & hls.BlockParameterTypes.NNR_CPT_DC != 0:
       ndu_oob["decomposition_rank"] = decomposition_parameter_dict["decomposition_rank"]
       ndu_oob["g_number_of_rows"] = decomposition_parameter_dict["g_number_of_rows"]
   return ndu_oob
    
    
def compile_ndu(param, approx_data, enc_info, model_info, ndu_oob, is_block, cpt, block_access, tensor_dims=None):
    ndu_header = {}
    ndu_header.update( ndu_oob )

    if ndu_header["input_parameters_present_flag"] == 1: 
        if ndu_header["tensor_dimensions_flag"] == 1:
            assert tensor_dims is not None, "tensor_dimensions must be specified!"
            ndu_header["count_tensor_dimensions"] = len(tensor_dims)
            ndu_header["tensor_dimensions"] = tensor_dims
        if ndu_header["cabac_unary_length_flag"] == 1:
            ndu_header["cabac_unary_length_minus1"] = enc_info["cabac_unary_length_minus1"]
        ndu_header["compressed_parameter_types"] = cpt
        if cpt & hls.BlockParameterTypes.NNR_CPT_DC != 0:
            ndu_header["decomposition_rank"] = approx_data["decomposition_rank"][block_access.block_id]
            ndu_header["g_number_of_rows"]   = approx_data["g_number_of_rows"][block_access.block_id]
    else:
        raise NotImplementedError("Out-of-band handling of input parameters not yet implemented!")
        
    ndu_header["mps_topology_indexed_reference_flag"] = 0 ##REQUIRED DURING CODING

    # nnr_unit_header syntax elements
    ndu_header["nnr_unit_type"] = hls.NnrUnitType.NNR_NDU
    ndu_header["partial_data_counter_present_flag"] = 0 ##tbd: set correctly elsewhere
    ndu_header["partial_data_counter"] = 0
    ndu_header["independently_decodable_flag"] = 1

    # compressed_data_unit_header syntax elements
    if is_block:
        ndu_header["nnr_compressed_data_unit_payload_type"] = hls.CompressedDataUnitPayloadType.NNR_PT_BLOCK
        assert block_access != None , "Block access undefined"
        if cpt & hls.BlockParameterTypes.NNR_CPT_DC != 0:
            param   = block_access.dc_g
            param_h = block_access.dc_h
        else:
            param = block_access.w
    elif (approx_data["approx_method"][param] == "uniform") or (approx_data["approx_method"][param] == "codebook"):
        ndu_header["nnr_compressed_data_unit_payload_type"] = hls.CompressedDataUnitPayloadType.NNR_PT_FLOAT
    elif approx_data["approx_method"][param] == "skip":
        ndu_header["nnr_compressed_data_unit_payload_type"] = hls.CompressedDataUnitPayloadType.NNR_PT_INT
    else:
        assert param not in approx_data["approx_method"], "Unsupported approx_method."
        ndu_header["nnr_compressed_data_unit_payload_type"] = hls.CompressedDataUnitPayloadType.NNR_PT_RAW_FLOAT
        ndu_header["raw_float32_parameter"] = approx_data["parameters"][param]
        
    assert param != None 

    if (
        (ndu_header["nnr_compressed_data_unit_payload_type"] == hls.CompressedDataUnitPayloadType.NNR_PT_BLOCK) or
        (ndu_header["nnr_compressed_data_unit_payload_type"] == hls.CompressedDataUnitPayloadType.NNR_PT_FLOAT) or
        (ndu_header["nnr_compressed_data_unit_payload_type"] == hls.CompressedDataUnitPayloadType.NNR_PT_INT)
    ):
        ndu_header["dq_flag"] = approx_data["dq_flag"][param]

    ndu_header["nnr_multiple_topology_elements_present_flag"] = 1 if ndu_header["nnr_compressed_data_unit_payload_type"] == hls.CompressedDataUnitPayloadType.NNR_PT_BLOCK else 0
    ndu_header["nnr_decompressed_data_format_present_flag"] = 0

    if ndu_header["nnr_multiple_topology_elements_present_flag"] == 1:
        tpl_elem_ids = [p for p in block_access.topology_elem_generator(approx_data["compressed_parameter_types"])]
        if ndu_header["mps_topology_indexed_reference_flag"] == 1:
            assert 0, "topology indexed not implemented yet!"
        else:
            ndu_header["count_topology_elements_minus2"] = len(tpl_elem_ids) - 2
            ndu_header["topology_elem_id_list"] = tpl_elem_ids
    else:
        if ndu_header["mps_topology_indexed_reference_flag"] == 1:
            assert 0, "topology indexed not implemented yet!"
        ndu_header["topology_elem_id"] = param #topology_elem_id == param Name

    if approx_data["approx_method"][param] == "codebook":
        ndu_header["codebook_present_flag"] = 1
        ndu_header["codebook_egk__"] = approx_data["codebooks_egk"][param]
        codebook_size = len( approx_data["codebooks"][param] )
        ndu_header["CbZeroOffset__"] = approx_data["codebook_zero_offsets"][param]
        ndu_header["codebook_size__"] = codebook_size
        ndu_header["codebook__"] = approx_data["codebooks"][param]
        if is_block and (cpt & hls.BlockParameterTypes.NNR_CPT_DC != 0):
            assert approx_data["approx_method"][param_h] == "codebook", "Params must have the same approx_method!"
            ndu_header["codebook_egk__dc"] = approx_data["codebooks_egk"][param_h]
            codebook_size = len( approx_data["codebooks"][param_h] )
            ndu_header["CbZeroOffset__dc"] = approx_data["codebook_zero_offsets"][param_h]
            ndu_header["codebook_size__dc"] = codebook_size
            ndu_header["codebook__dc"] = approx_data["codebooks"][param_h]
    else:
        ndu_header["codebook_present_flag"] = 0

    ndu_header["nnr_decompressed_data_format"] = hls.DecompressedDataFormat.TENSOR_FLOAT32

    if len( ndu_header["tensor_dimensions"] ) > 1:
        ndu_header["scan_order"] = approx_data["scan_order"][param]

    return ndu_header


def compile_ndu_eps( ndu_header, cabac_entry_point_list ):
    dims = ndu_header["tensor_dimensions"]
    height = dims[0]
    width = np.prod(dims[1:])
    if width > 1 and height > 1:
#    if len( ndu_header["tensor_dimensions"] ) > 1:
        if ndu_header["scan_order"] > 0:
#            ndu_header["cabac_entry_point_list_size"] = len(cabac_entry_point_list)
            ndu_header["cabac_entry_point_list"] = cabac_entry_point_list

    return ndu_header

def compile_tpl(model_info):
    if model_info["topology_storage_format"] == nnr_model.TopologyStorageFormat.NNR_TPL_UNREC: ##ignore topology_data in this case!
        tpl = {
            "topology_data": "",
            "topology_storage_format": nnr_model.TopologyStorageFormat.NNR_TPL_UNREC
        }
    elif model_info["topology_storage_format"] == nnr_model.TopologyStorageFormat.NNR_TPL_PYT: ##topology_data can be ignored. The parameter names are transmitted as topology_elem_id/topology_elem_id_index anyway!
        topology_data = '' 
        tpl = {
            "topology_data": topology_data,
            "topology_storage_format": nnr_model.TopologyStorageFormat.NNR_TPL_PYT
        }
    elif model_info["topology_storage_format"] == nnr_model.TopologyStorageFormat.NNR_TPL_TEF: ##topology_data can be ignored. The parameter names are transmitted as topology_elem_id/topology_elem_id_index anyway!
        topology_data = ''
        tpl = {
            "topology_data": topology_data,
            "topology_storage_format": nnr_model.TopologyStorageFormat.NNR_TPL_TEF
        }
    else:
        raise NotImplementedError("The given topology storage format is not implemented.")

    tpl.update({
            "nnr_unit_type": hls.NnrUnitType.NNR_TPL,
            "partial_data_counter_present_flag": 0,
            "partial_data_counter": 0,
            "independently_decodable_flag": 1,
            "topology_compression_format" : model_info["topology_compression_format"]
    })
    return tpl


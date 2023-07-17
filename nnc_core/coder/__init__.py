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
import copy
from . import syntax_compiler, baseline
from nnc_core import hls
from nnc_core import nnr_model
from .. import common
import deepCABAC
from nnc_core.nnr_model import NNRModelAccess

def is_block_possible(block_access, approx_data):
    # disable block if decomposed tensors use different approx_methods
    if block_access.dc_g in approx_data['approx_method']:
        if approx_data['approx_method'][block_access.dc_g] != approx_data['approx_method'][block_access.dc_h]:
            return False

    # disable block if non-weights use codebook
    for par_type, param, _ in block_access.param_generator(approx_data["compressed_parameter_types"]):
        if not par_type.endswith("weight"):
            if approx_data['approx_method'][param] == "codebook":
                return False
            
    # disable block if parameters have inconsistend dq_flags    
    dq_flag = -1
    for _, param, _ in block_access.param_generator(approx_data["compressed_parameter_types"]):
        if dq_flag == -1:
            dq_flag = approx_data["dq_flag"][param]
        else:
            if approx_data["dq_flag"][param] != dq_flag:
                print( "Disabled DQ for block because of inconsistend dq_flags.")
                return False
    
    # disable block if parameters are of type integer
    for par_type, param, _ in block_access.param_generator(approx_data["compressed_parameter_types"]):
        if approx_data['approx_method'][param] == "skip":
            return False
    
    return True


def ndu_enc_generator(enc_info, model_info, approx_data):
    model_access = NNRModelAccess(model_info)
    for block_or_param in model_access.blocks_and_params():
        block_id = block_or_param.block_id
        if block_id is None:
            param = block_or_param.param
            ndu_oob = syntax_compiler.compile_ndu_oob() ##Only in-band-signaling
            ib_dims = approx_data["parameters"][param].shape
            assert ib_dims is not None
            ndu = syntax_compiler.compile_ndu(param, approx_data, enc_info, model_info, ndu_oob, False, 0, None, ib_dims)
            yield ndu, [param]
        else:
            cpt = approx_data["compressed_parameter_types"][block_id]
            if is_block_possible( block_or_param, approx_data):
                ndu_oob = syntax_compiler.compile_ndu_oob() ##No Parameters specified -> No oob signaling
                ndu = syntax_compiler.compile_ndu(None, approx_data, enc_info, model_info, ndu_oob, True, cpt, block_or_param, model_info["parameter_dimensions"][block_or_param.w])
                yield ndu, [x for _, x, _ in block_or_param.param_generator(approx_data["compressed_parameter_types"])]
            else:
                for _, param, dims in block_or_param.param_generator(approx_data["compressed_parameter_types"]):
                    ndu_oob = syntax_compiler.compile_ndu_oob() ##No oob signaling    
                    ndu = syntax_compiler.compile_ndu(param, approx_data, enc_info, model_info, ndu_oob, False, cpt, block_or_param, dims)
                    yield ndu, [param]

                
def __get_topology_elem_id_order(compressed_parameter_types):
    id_list = []

    if compressed_parameter_types & hls.BlockParameterTypes.NNR_CPT_LS:
        if compressed_parameter_types & hls.BlockParameterTypes.NNR_CPT_DC:
            id_list.append(2)
        else:
            id_list.append(1)
    if compressed_parameter_types & hls.BlockParameterTypes.NNR_CPT_BI:
        index = 1
        if compressed_parameter_types & hls.BlockParameterTypes.NNR_CPT_DC:
            index += 1
        if compressed_parameter_types & hls.BlockParameterTypes.NNR_CPT_LS:
            index += 1
        if compressed_parameter_types & hls.BlockParameterTypes.NNR_CPT_BN:
            index += 4
        id_list.append(index)
    if compressed_parameter_types & hls.BlockParameterTypes.NNR_CPT_BN:
        index = 1
        if compressed_parameter_types & hls.BlockParameterTypes.NNR_CPT_DC:
            index += 1
        if compressed_parameter_types & hls.BlockParameterTypes.NNR_CPT_LS:
            index += 1
        id_list.append(index)
        id_list.append(index+1)
        id_list.append(index+2)
        id_list.append(index+3)

    id_list.append(0)
    if compressed_parameter_types & hls.BlockParameterTypes.NNR_CPT_DC:
        id_list.append(1)

    return id_list
                
    
def encode(enc_info, model_info, approx_data):
    ndu_start = syntax_compiler.compile_start_unit(0)
    bs = hls.encode_nnr_unit_with_size_dummy( ndu_start )
    bs, _ = hls.update_nnr_unit_size( bs )
    mps = syntax_compiler.compile_mps( approx_data, "topology_storage_format" in model_info)
    lps = None
    bs_mps = hls.encode_nnr_unit_with_size_dummy( mps )
    bs_mps, _ = hls.update_nnr_unit_size( bs_mps )
    bs.extend(bs_mps)

    
    if model_info["topology_storage_format"] is not None:
        tpl = syntax_compiler.compile_tpl( model_info )
        bs_tpl = hls.encode_nnr_unit_with_size_dummy( tpl )
        bs_tpl, _ = hls.update_nnr_unit_size( bs_tpl )
        bs.extend(bs_tpl)
    
    for ndu, params in ndu_enc_generator(enc_info, model_info, approx_data):
        encoder = deepCABAC.Encoder()
        num_coded_params = 0
        for param in params:
            if param in approx_data['approx_method']:
                baseline.encode(encoder, approx_data, param, ndu, mps, lps, enc_info['param_opt_flag'])
                num_coded_params += 1

        bs_par = bytearray( encoder.finish().tobytes() )

        decoder = deepCABAC.Decoder()
        decoder.setStream( bs_par )
        approx_data_ep = copy.deepcopy( approx_data )

        epList = np.array([], dtype=np.uint64)

        for param in params:
            if param in approx_data_ep['approx_method']:
                if param in approx_data["scan_order"] and approx_data["scan_order"][param] > 0:
                    approx_data_ep["parameters"][param] =  np.zeros_like(approx_data["parameters"][param], dtype=np.int32)  
                    epListPart = baseline.decodeAndCreateEPs(decoder, approx_data_ep, param, ndu, mps, lps)
                    epList = np.concatenate([epList, epListPart])

        ndu = syntax_compiler.compile_ndu_eps( ndu, epList )
      
        bs_ndu = hls.encode_nnr_unit_with_size_dummy(ndu)
        if num_coded_params > 0:
            bs_ndu.extend( bs_par )
        bs_ndu, _ = hls.update_nnr_unit_size(bs_ndu)
        bs.extend( bs_ndu )

    return bs
    

def __decode_nnr_start_unit(nnr_gen, ndu_start, hls_stats = {}):
    next(nnr_gen) # resume to final yield
    bytes_start = next(nnr_gen)
    assert bytes_start == ndu_start["nnr_unit_size"], "nnr_unit_size doesn't match the number of decoded bytes."
    hls_stats["start_bytes"] = bytes_start
    return bytes_start

def __decode_nnr_mps_unit(nnr_gen, reader, ndu_mps, hls_stats = {}):
    next(nnr_gen) # final yield of nnr_unit_size_and_header
    hls.decode_nnr_unit_payload(reader, ndu_mps)
    bytes_ndu = reader.getNumBytesTouched()
    assert bytes_ndu == ndu_mps["nnr_unit_size"], "nnr_unit_size doesn't match the number of decoded bytes."
    hls_stats["mps_bytes"] = bytes_ndu
    return bytes_ndu

def __decode_nnr_lps_unit(nnr_gen, reader, ndu_lps, hls_stats = {}):
    next(nnr_gen) # final yield of nnr_unit_size_and_header
    hls.decode_nnr_unit_payload(reader, ndu_lps)
    bytes_ndu = reader.getNumBytesTouched()
    assert bytes_ndu == ndu_lps["nnr_unit_size"], "nnr_unit_size doesn't match the number of decoded bytes."
    hls_stats["lps_bytes"] = bytes_ndu
    return bytes_ndu

def __decode_nnr_tpl_unit(nnr_gen, reader, ndu_tpl, mps, model_info, hls_stats = {}):
    next(nnr_gen) # final yield of nnr_unit_size_and_header
    hls.decode_nnr_unit_payload(reader, ndu_tpl)
    bytes_ndu = reader.getNumBytesTouched()
    assert bytes_ndu == ndu_tpl["nnr_unit_size"], "nnr_unit_size doesn't match the number of decoded bytes."
    hls_stats["bytes_tpl"] = bytes_ndu
    model_info["topology_storage_format"] = ndu_tpl["topology_storage_format"]
    if ndu_tpl["topology_storage_format"] == nnr_model.TopologyStorageFormat.NNR_TPL_UNREC:
        pass
    elif ndu_tpl["topology_storage_format"] == nnr_model.TopologyStorageFormat.NNR_TPL_NNEF:
        raise NotImplementedError("NNEF Topology Storage Format not yet implemented!")
    elif ndu_tpl["topology_storage_format"] == nnr_model.TopologyStorageFormat.NNR_TPL_ONNX:
        raise NotImplementedError("ONNX Topology Storage Format not yet implemented!")
    elif ndu_tpl["topology_storage_format"] == nnr_model.TopologyStorageFormat.NNR_TPL_PYT:
        pass
    elif ndu_tpl["topology_storage_format"] == nnr_model.TopologyStorageFormat.NNR_TPL_TEF:
        pass
    elif ndu_tpl["topology_storage_format"] == nnr_model.TopologyStorageFormat.NNR_TPL_PRUN:
        raise NotImplementedError("PRUN Topology Storage Format not yet implemented!")
    elif ndu_tpl["topology_storage_format"] == nnr_model.TopologyStorageFormat.NNR_TPL_REFLIST:
        raise NotImplementedError("REFLIST Topology Storage Format not yet implemented!")
    return bytes_ndu

def __decode_nnr_qnt_unit():
    pass

def __decode_nnr_ndu_unit(nnr_gen, reader, bitstream, ndu, mps, lps, tpl, model_info, approx_data, bytes_read, decoded_dc_tensorG, set_model_info=True, hls_stats = {}):
    block_id = None
    parameter_index = len(model_info["parameter_index"].keys())
    add_block_id_to_model_info = False
    #model_info["block_identifier"] = {}
    
    ndu.update({ "mps_topology_indexed_reference_flag": mps["mps_topology_indexed_reference_flag"]})
    
    next(nnr_gen) # continue after decoding nnr unit type and stop after input_parameters_present_flag
    
    cpt = ndu["compressed_parameter_types"]
    if decoded_dc_tensorG:
        assert cpt & hls.BlockParameterTypes.NNR_CPT_DC != 0, "Preceding NDU contained a decomposed tensor G! This NDU must contain tensor H!"
    if ndu["nnr_compressed_data_unit_payload_type"] != hls.CompressedDataUnitPayloadType.NNR_PT_BLOCK: 
        parType = "unspecified"
        if ndu["nnr_multiple_topology_elements_present_flag"]:
            raise NotImplementedError("Tensor Split not yet implemented!")
        else:
            if not ndu["mps_topology_indexed_reference_flag"]:
                param = ndu["topology_elem_id"]
            else:
                assert tpl["topology_storage_format"] == nnr_model.TopologyStorageFormat.NNR_TPL_REFLIST, "TPL Unit must be of type NNR_TPL_REFLIST!"
                raise NotImplementedError("REFLIST Topology storage format not yet implemented!")

        next(nnr_gen) # continue decoding of the nnr unit size and header after input_parameters_present_flag
        if ndu["partial_data_counter_present_flag"] == 1:
            assert 0, "partial_data_counter not yet implemented!"
        hls.decode_nnr_unit_payload(reader, ndu)
        bytes_ndu = reader.getNumBytesTouched()

        if cpt & hls.BlockParameterTypes.NNR_CPT_DC != 0:
            add_block_id_to_model_info = param[:-2] not in model_info["block_identifier"]
            parType = "weight"
            if not decoded_dc_tensorG and param.endswith("_G"): ##param.endwith is optional!
                ndu["_decomposed_tensor_type"] = "G"
                decoded_dc_tensorG = True
            elif decoded_dc_tensorG and param.endswith("_H"):
                ndu["_decomposed_tensor_type"] = "H"
                decoded_dc_tensorG = False
        else:
            if param in model_info["block_identifier"]:
                del model_info["block_identifier"][param]

        params = [(parType, param, ndu["tensor_dimensions"])]
    else:
        assert ndu["nnr_compressed_data_unit_payload_type"] == hls.CompressedDataUnitPayloadType.NNR_PT_BLOCK, "Payload Type must be NNR_PT_BLOCK"
        assert ndu["nnr_multiple_topology_elements_present_flag"], "nnr_multiple_topology_elements_present_flag must be euqal to one for NNR_PT_BLOCK"
        
        block_id = None
        params = []
        id_list_counter = 0
        id_index_list = __get_topology_elem_id_order(cpt)

        ##Get tensor_dimensions and count_tensor_dimensions if neccessary (If they are not transmitted in the bitstream)
        if not ndu["mps_topology_indexed_reference_flag"]:
            weight_param = ndu["topology_elem_id_list"][id_index_list[-1]]
        else:
            assert tpl["topology_storage_format"] == nnr_model.TopologyStorageFormat.NNR_TPL_REFLIST, "TPL Unit must be of type NNR_TPL_REFLIST!"
            raise NotImplementedError("REFLIST Topology storage format not yet implemented!")
        if weight_param.endswith("_G") or weight_param.endswith("_H"):
            weight_param = weight_param[:-2]

        add_block_id_to_model_info = weight_param not in model_info["block_identifier"]

        next(nnr_gen) # continue decoding of the nnr unit size and header 
        if ndu["partial_data_counter_present_flag"] == 1:
            assert 0, "partial_data_counter not yet implemented!"
        hls.decode_nnr_unit_payload(reader, ndu)
        bytes_ndu = reader.getNumBytesTouched()

        if cpt & hls.BlockParameterTypes.NNR_CPT_LS != 0:
            parType = "weight.ls"
            if not ndu["mps_topology_indexed_reference_flag"]:
                param = ndu["topology_elem_id_list"][id_index_list[id_list_counter]]
            else:
                assert tpl["topology_storage_format"] == nnr_model.TopologyStorageFormat.NNR_TPL_REFLIST, "TPL Unit must be of type NNR_TPL_REFLIST!"
                topology_elem_id_list_index = ndu["topology_elem_id_index_list"][id_index_list[id_list_counter]]
                param = model_info["topology_elem_id_list"][topology_elem_id_list_index]
            id_list_counter += 1
            dims = [ndu["tensor_dimensions"][0]]
            params.append( (parType, param, dims) )

        if cpt & hls.BlockParameterTypes.NNR_CPT_BI != 0:
            parType = "bias"
            if not ndu["mps_topology_indexed_reference_flag"]:
                param = ndu["topology_elem_id_list"][id_index_list[id_list_counter]]
            else:
                assert tpl["topology_storage_format"] == nnr_model.TopologyStorageFormat.NNR_TPL_REFLIST, "TPL Unit must be of type NNR_TPL_REFLIST!"
                topology_elem_id_list_index = ndu["topology_elem_id_index_list"][id_index_list[id_list_counter]]
                param = model_info["topology_elem_id_list"][topology_elem_id_list_index]
            id_list_counter += 1
            dims = [ndu["tensor_dimensions"][0]]
            params.append( (parType, param, dims) )
        
        if cpt & hls.BlockParameterTypes.NNR_CPT_BN != 0:
            parType = "bn."
            dims = [ndu["tensor_dimensions"][0]]
            for bn_param in ["beta", "gamma", "mean", "var"]:
                if not ndu["mps_topology_indexed_reference_flag"]:
                    param  = ndu["topology_elem_id_list"][id_index_list[id_list_counter]]
                else:
                    assert tpl["topology_storage_format"] == nnr_model.TopologyStorageFormat.NNR_TPL_REFLIST, "TPL Unit must be of type NNR_TPL_REFLIST!"
                    topology_elem_id_list_index = ndu["topology_elem_id_index_list"][id_index_list[id_list_counter]]
                    param  = model_info["topology_elem_id_list"][topology_elem_id_list_index]
                id_list_counter += 1
                params.append( (parType+bn_param , param , dims) )

        if cpt & hls.BlockParameterTypes.NNR_CPT_DC == 0:
            parType = "weight"
            if not ndu["mps_topology_indexed_reference_flag"]:
                param = ndu["topology_elem_id_list"][id_index_list[id_list_counter]]
            else:
                assert tpl["topology_storage_format"] == nnr_model.TopologyStorageFormat.NNR_TPL_REFLIST, "TPL Unit must be of type NNR_TPL_REFLIST!"
                topology_elem_id_list_index = ndu["topology_elem_id_index_list"][id_index_list[id_list_counter]]
                param = model_info["topology_elem_id_list"][topology_elem_id_list_index]
            id_list_counter += 1
            dims = ndu["tensor_dimensions"]     
            params.append( (parType, param, dims) )
        
        if cpt & hls.BlockParameterTypes.NNR_CPT_DC != 0:
            parType = "weight"
            for dc_params in ["_G", "_H"]:
                if not ndu["mps_topology_indexed_reference_flag"]:
                    param = ndu["topology_elem_id_list"][id_index_list[id_list_counter]]
                else:
                    assert tpl["topology_storage_format"] == nnr_model.TopologyStorageFormat.NNR_TPL_REFLIST, "TPL Unit must be of type NNR_TPL_REFLIST!"
                    topology_elem_id_list_index = ndu["topology_elem_id_index_list"][id_index_list[id_list_counter]]
                    param = model_info["topology_elem_id_list"][topology_elem_id_list_index]
                id_list_counter += 1
                dims = ndu["tensor_dimensions"]
                params.append( (parType, param, dims) )
        
        assert id_list_counter == ndu["count_topology_elements_minus2"] + 2, "Number of decoded topology elements does not match count_topology_elements_minus2 + 2!"
    
    for par_type, param, dims in params:
        if param.endswith("_G") or param.endswith("_H"):
            param = param[:-2]
        if param in model_info["block_identifier"]:
            block_id = model_info["block_identifier"][param]
        elif add_block_id_to_model_info:
            block_id = weight_param
            model_info["block_identifier"][param] = block_id
        if param not in model_info["parameter_dimensions"] and set_model_info:
            model_info["parameter_dimensions"][param] = dims
        if param not in model_info["parameter_type"] and set_model_info:
            model_info["parameter_type"][param] = par_type
        if param not in model_info["parameter_index"] and set_model_info:
            model_info["parameter_index"][param] = parameter_index
        parameter_index += 1 

    if block_id:
        approx_data["compressed_parameter_types"][block_id] = cpt


    hls_stats["ndu_bytes"].append( bytes_ndu )
    if block_id is not None:
        assert approx_data["compressed_parameter_types"][block_id] == cpt
    decoder = deepCABAC.Decoder()
    decoder_initialized = False
    for par_type, param, dims in params:
        if ndu["nnr_compressed_data_unit_payload_type"] == hls.CompressedDataUnitPayloadType.NNR_PT_RAW_FLOAT:
            assert param not in approx_data["approx_method"]
            approx_data["parameters"][param] = ndu["raw_float32_parameter"]
            hls_stats["ndu_bytes"][-1] -= 4 * ndu["raw_float32_parameter"].size
        else:
            if (
                (ndu["nnr_compressed_data_unit_payload_type"] == hls.CompressedDataUnitPayloadType.NNR_PT_BLOCK) and 
                (ndu["compressed_parameter_types"] & hls.BlockParameterTypes.NNR_CPT_DC != 0) and
                (ndu["codebook_present_flag"] == 1)
            ):
                if param.endswith("_G"):
                    approx_data["approx_method"][param]         = 'codebook'
                    approx_data["codebooks"][param]             = ndu["codebook__"]
                    approx_data["codebook_zero_offsets"][param] = ndu["CbZeroOffset__"]
                    approx_data["codebooks_egk"][param]         = ndu["codebook_egk__"]       
                elif param.endswith("_H"):
                    approx_data["approx_method"][param]         = 'codebook'
                    approx_data["codebooks"][param]             = ndu["codebook__dc"]
                    approx_data["codebook_zero_offsets"][param] = ndu["CbZeroOffset__dc"]
                    approx_data["codebooks_egk"][param]         = ndu["codebook_egk__dc"]

            elif (
                  (ndu["nnr_compressed_data_unit_payload_type"] == hls.CompressedDataUnitPayloadType.NNR_PT_BLOCK) and
                  (ndu["codebook_present_flag"] == 1) and
                  (par_type.endswith("weight"))
            ): 
                approx_data["approx_method"][param]         = 'codebook'
                approx_data["codebooks"][param]             = ndu["codebook__"]
                approx_data["codebook_zero_offsets"][param] = ndu["CbZeroOffset__"] 
                approx_data["codebooks_egk"][param]         = ndu["codebook_egk__"]                       
            elif(
                 (ndu["nnr_compressed_data_unit_payload_type"] != hls.CompressedDataUnitPayloadType.NNR_PT_BLOCK) and
                 (ndu.get("codebook_present_flag") == 1)
            ): 
                approx_data["approx_method"][param]         = 'codebook'
                approx_data["codebooks"][param]             = ndu["codebook__"]
                approx_data["codebook_zero_offsets"][param] = ndu["CbZeroOffset__"] 
                approx_data["codebooks_egk"][param]         = ndu["codebook_egk__"]
            elif ndu["nnr_compressed_data_unit_payload_type"] == hls.CompressedDataUnitPayloadType.NNR_PT_INT :
                approx_data["approx_method"][param] = 'skip'
            else:
                approx_data["approx_method"][param] = 'uniform'


            if ndu["count_tensor_dimensions"] > 1:
                approx_data["scan_order"][param] = ndu['scan_order']
                if ndu['scan_order'] > 0:
                    tensorDimensions   = dims
                    blockDim           = 4 << ndu['scan_order']
                    
                    if ndu["compressed_parameter_types"] & hls.BlockParameterTypes.NNR_CPT_DC != 0:
                        hNumberOfColumns  = np.int32(np.prod( tensorDimensions )/ndu["g_number_of_rows"])
                        tensorDimensionsG = [ndu["g_number_of_rows"], ndu["decomposition_rank"]] 
                        tensorDimensionsH = [ndu["decomposition_rank"], hNumberOfColumns]
                    
                    if (
                        (ndu["nnr_compressed_data_unit_payload_type"] != hls.CompressedDataUnitPayloadType.NNR_PT_BLOCK) and
                        (ndu["compressed_parameter_types"] & hls.BlockParameterTypes.NNR_CPT_DC != 0)
                    ):
                        if ndu["_decomposed_tensor_type"] == "G":
                            tensorDimensions = tensorDimensionsG
                        elif ndu["_decomposed_tensor_type"] == "H":
                            tensorDimensions = tensorDimensionsH

                    numBlockRowsMinus1 = ((tensorDimensions[0]+blockDim-1) >> (2+ndu['scan_order'])) - 1
                    entryPoints = ndu["cabac_entry_point_list"][0:numBlockRowsMinus1]

                    if (
                        (ndu["nnr_compressed_data_unit_payload_type"] == hls.CompressedDataUnitPayloadType.NNR_PT_BLOCK) and
                        (ndu["compressed_parameter_types"] & hls.BlockParameterTypes.NNR_CPT_DC != 0)
                    ):
                        numBlockRowsMinus1G  = ((tensorDimensionsG[0]+blockDim-1) >> (2+ndu["scan_order"])) - 1
                        numBlockRowsMinus1H  = ((tensorDimensionsH[0]+blockDim-1) >> (2+ndu["scan_order"])) - 1

                        if param.endswith("_G"):
                            entryPoints =  ndu["cabac_entry_point_list"][0:numBlockRowsMinus1G]
                        elif param.endswith("_H"):
                            entryPoints =  ndu["cabac_entry_point_list"][numBlockRowsMinus1G:(numBlockRowsMinus1G+numBlockRowsMinus1H)]

                    decoder.setEntryPoints( entryPoints )


            tensorDimensions = dims

            if ndu["compressed_parameter_types"] & hls.BlockParameterTypes.NNR_CPT_DC != 0:
                hNumberOfColumns  = np.int32(np.prod( tensorDimensions )/ndu["g_number_of_rows"])
                tensorDimensionsG = [ndu["g_number_of_rows"], ndu["decomposition_rank"]] 
                tensorDimensionsH = [ndu["decomposition_rank"], hNumberOfColumns]

            if param.endswith("_G"):
                dims = tensorDimensionsG
            elif param.endswith("_H"):
                dims = tensorDimensionsH

            approx_data["parameters"][param] = np.zeros(dims, dtype=np.int32)
            if not decoder_initialized:
                decoder.setStream( bitstream[bytes_read+bytes_ndu:] )
                decoder_initialized = True
            baseline.decode(decoder, approx_data, param, ndu, mps, lps)

        if lps is not None:
            model_info["performance_map_flags"]["lps_sparsification_flag"][param]                = lps["lps_sparsification_flag"]
            model_info["performance_map_flags"]["lps_pruning_flag"][param]                       = lps["lps_pruning_flag"]
            model_info["performance_map_flags"]["lps_unification_flag"][param]                   = lps["lps_unification_flag"]
            model_info["performance_map_flags"]["lps_decomposition_performance_map_flag"][param] = lps["lps_decomposition_performance_map_flag"]
        else:
            model_info["performance_map_flags"]["lps_sparsification_flag"][param]                = 0
            model_info["performance_map_flags"]["lps_pruning_flag"][param]                       = 0
            model_info["performance_map_flags"]["lps_unification_flag"][param]                   = 0
            model_info["performance_map_flags"]["lps_decomposition_performance_map_flag"][param] = 0
        
        if mps is not None:
            model_info["performance_map_flags"]["mps_sparsification_flag"][param]                = mps["mps_sparsification_flag"]
            model_info["performance_map_flags"]["mps_pruning_flag"][param]                       = mps["mps_pruning_flag"]
            model_info["performance_map_flags"]["mps_unification_flag"][param]                   = mps["mps_unification_flag"]
            model_info["performance_map_flags"]["mps_decomposition_performance_map_flag"][param] = mps["mps_decomposition_performance_map_flag"]
        else:
            model_info["performance_map_flags"]["mps_sparsification_flag"][param]                = 0
            model_info["performance_map_flags"]["mps_pruning_flag"][param]                       = 0
            model_info["performance_map_flags"]["mps_unification_flag"][param]                   = 0
            model_info["performance_map_flags"]["mps_decomposition_performance_map_flag"][param] = 0
        
    if decoder_initialized:
        bytes_ndu += decoder.finish()
    assert bytes_ndu == ndu["nnr_unit_size"], "nnr_unit_size doesn't match the number of decoded bytes."

    return bytes_ndu, decoded_dc_tensorG


def __decode_nnr_unit(reader, bitstream, bytes_read, mps, lps, tpl, model_info, approx_data, nnr_ndu_decoded, decoded_dc_tensorG, set_model_info, hls_stats={}):
    bytes_ndu = 0
    ndu = {}
    g = hls.decode_nnr_unit_size_and_header(reader, ndu)
    next(g) # start decoding of the nnr unit size and header and stop at nnr unit type
    
    if decoded_dc_tensorG:
        assert ndu["nnr_unit_type"] == hls.NnrUnitType.NNR_NDU, "Preceding NDU contained a decomposed tensor G! This NDU must be of type NNR_NDU!"
    
    if ndu["nnr_unit_type"] == hls.NnrUnitType.NNR_MPS:
        assert mps is None, "NNR_MPS Unit already decoded! There shall be only one NNR_MPS in the bitstream!"
        assert not nnr_ndu_decoded, "An NNR_MPS shall precede any NNR_NDU, but an NNR_NDU has already been decoded!"       
        bytes_ndu = __decode_nnr_mps_unit( g, reader, ndu, hls_stats )
        mps = ndu
        if "mps_qp_density" in mps:
            approx_data.update( { "qp" : {}, "qp_density" : np.int32(mps["mps_qp_density"]), "dq_flag" : {} } )

        if mps["mps_sparsification_flag"] == 1 :
            model_info["performance_maps"]["mps"]["sparsification_performance_map"] = {
                "spm_count_thresholds":     mps["spm_count_thresholds"],
                "sparsification_threshold": mps["sparsification_threshold"],
                "non_zero_ratio":           mps["non_zero_ratio"],
                "spm_nn_accuracy":          mps["spm_nn_accuracy"],
                "spm_count_classes":        mps["spm_count_classes"],
                "spm_class_bitmask":        mps["spm_class_bitmask"],
                "spm_nn_class_accuracy":    mps["spm_nn_class_accuracy"]
            }
        if mps["mps_pruning_flag"] == 1 :
            model_info["performance_maps"]["mps"]["pruning_performance_map"] = {
                "ppm_count_pruning_ratios": mps["ppm_count_pruning_ratios"],
                "pruning_ratio":            mps["pruning_ratio"],
                "ppm_nn_accuracy":          mps["ppm_nn_accuracy"],
                "ppm_count_classes":        mps["ppm_count_classes"],
                "ppm_class_bitmask":        mps["ppm_class_bitmask"],
                "ppm_nn_class_accuracy":    mps["ppm_nn_class_accuracy"]
            }
        if mps["mps_unification_flag"] == 1 :
            model_info["performance_maps"]["mps"]["unification_performance_map"] = {
                "upm_count_thresholds":            mps["upm_count_thresholds"],
                "count_reshaped_tensor_dimension": mps["count_reshaped_tensor_dimension"],
                "reshaped_tensor_dimensions":      mps["reshaped_tensor_dimensions"],
                "count_super_block_dimension":     mps["count_super_block_dimension"],
                "super_block_dimensions":          mps["super_block_dimensions"],
                "count_block_dimension":           mps["count_block_dimension"],
                "block_dimensions":                mps["block_dimensions"],
                "unification_threshold":           mps["unification_threshold"],
                "upm_nn_accuracy":                 mps["upm_nn_accuracy"],
                "upm_count_classes":               mps["upm_count_classes"],
                "upm_class_bitmask":               mps["upm_class_bitmask"],
                "upm_nn_class_accuracy":           mps["upm_nn_class_accuracy"]
            }
        if mps["mps_decomposition_performance_map_flag"] == 1 :
            model_info["performance_maps"]["mps"]["decomposition_performance_map"] = {
                "dpm_count_thresholds":  mps["dpm_count_thresholds"],
                "mse_threshold":         mps["mse_threshold"],
                "dpm_nn_accuracy":       mps["dpm_nn_accuracy"],
                "nn_reduction_ratio":    mps["nn_reduction_ratio"],
                "dpm_count_classes":     mps["dpm_count_classes"],
                "dpm_nn_class_accuracy": mps["dpm_nn_class_accuracy"]
            }

    elif ndu["nnr_unit_type"] == hls.NnrUnitType.NNR_LPS:
        bytes_ndu = __decode_nnr_lps_unit( g, reader, ndu, hls_stats )
        lps = ndu
        if "lps_qp_density" in lps:
            approx_data.update( { "qp" : {}, "qp_density" : np.int32(lps["lps_qp_density"]), "dq_flag" : {} } )

        if lps["lps_sparsification_flag"] == 1 :
            model_info["performance_maps"]["lps"]["sparsification_performance_map"] = {
                "spm_count_thresholds":     lps["spm_count_thresholds"],
                "sparsification_threshold": lps["sparsification_threshold"],
                "non_zero_ratio":           lps["non_zero_ratio"],
                "spm_nn_accuracy":          lps["spm_nn_accuracy"],
                "spm_count_classes":        lps["spm_count_classes"],
                "spm_class_bitmask":        lps["spm_class_bitmask"],
                "spm_nn_class_accuracy":    lps["spm_nn_class_accuracy"]
            }
        if lps["lps_pruning_flag"] == 1 :
            model_info["performance_maps"]["lps"]["pruning_performance_map"] = {
                "ppm_count_pruning_ratios": lps["ppm_count_pruning_ratios"],
                "pruning_ratio":            lps["pruning_ratio"],
                "ppm_nn_accuracy":          lps["ppm_nn_accuracy"],
                "ppm_count_classes":        lps["ppm_count_classes"],
                "ppm_class_bitmask":        lps["ppm_class_bitmask"],
                "ppm_nn_class_accuracy":    lps["ppm_nn_class_accuracy"]
            }
        if lps["lps_unification_flag"] == 1 :
            model_info["performance_maps"]["lps"]["unification_performance_map"] = {
                "upm_count_thresholds":            lps["upm_count_thresholds"],
                "count_reshaped_tensor_dimension": lps["count_reshaped_tensor_dimension"],
                "reshaped_tensor_dimensions":      lps["reshaped_tensor_dimensions"],
                "count_super_block_dimension":     lps["count_super_block_dimension"],
                "super_block_dimensions":          lps["super_block_dimensions"],
                "count_block_dimension":           lps["count_block_dimension"],
                "block_dimensions":                lps["block_dimensions"],
                "unification_threshold":           lps["unification_threshold"],
                "upm_nn_accuracy":                 lps["upm_nn_accuracy"],
                "upm_count_classes":               lps["upm_count_classes"],
                "upm_class_bitmask":               lps["upm_class_bitmask"],
                "upm_nn_class_accuracy":           lps["upm_nn_class_accuracy"]
            }
        if lps["lps_decomposition_performance_map_flag"] == 1 :
            model_info["performance_maps"]["lps"]["decomposition_performance_map"] = {
                "dpm_count_thresholds":  lps["dpm_count_thresholds"],
                "mse_threshold":         lps["mse_threshold"],
                "dpm_nn_accuracy":       lps["dpm_nn_accuracy"],
                "nn_reduction_ratio":    lps["nn_reduction_ratio"],
                "dpm_count_classes":     lps["dpm_count_classes"],
                "dpm_nn_class_accuracy": lps["dpm_nn_class_accuracy"]
            }

    elif ndu["nnr_unit_type"] == hls.NnrUnitType.NNR_TPL:
        bytes_ndu = __decode_nnr_tpl_unit( g, reader, ndu, mps, model_info, hls_stats )
        tpl = ndu

    elif ndu["nnr_unit_type"] == hls.NnrUnitType.NNR_QNT:
        raise NotImplementedError("Decoding of NNR_LPS Units not yet implemented!")

    elif ndu["nnr_unit_type"] == hls.NnrUnitType.NNR_AGG:
        raise NotImplementedError("Decoding of NNR_AGG Units not yet implemented!")
        
    elif ndu["nnr_unit_type"] == hls.NnrUnitType.NNR_NDU:
        bytes_ndu, decoded_dc_tensorG = __decode_nnr_ndu_unit(g, reader, bitstream, ndu, mps, lps, tpl, model_info, approx_data, bytes_read, decoded_dc_tensorG, set_model_info, hls_stats)

    else:
        assert 0, "nnr_unit_type: {} is not specified!".format(ndu["nnr_unit_type"])

    nnr_ndu_decoded = True

    return bytes_ndu, mps, lps, tpl, model_info, approx_data, nnr_ndu_decoded, decoded_dc_tensorG

def decode(bitstream, model_info, hls_stats = {} ):
    assert isinstance(bitstream, (bytearray, bytes))

    if not isinstance(bitstream, bytearray):
        bitstream = bytearray(bitstream)
    
    bytes_read = None

    hls_stats["ndu_bytes"] = []
    approx_data = {
        "approx_method": {},
        "parameters": {},
        "compressed_parameter_types": {},
        "scan_order": {},
        "codebooks": {},
        "codebooks_egk": {},
        "codebook_zero_offsets": {}
    }
    mps = None
    lps = None
    tpl = None
    nnr_ndu_decoded = False
    decoded_dc_tensorG = False ##Only required if DC but not NNR_PT_BLOCK
    set_model_info = False if len(model_info["parameter_type"]) != 0 else True

    ndu_start = {}
    nnr_gen = hls.decode_nnr_unit(bitstream, ndu_start)
    next(nnr_gen) # start decoding and stop at nnr unit type

    assert ndu_start["nnr_unit_type"] == hls.NnrUnitType.NNR_STR, "First nnr unit shall be of type NNR_STR."
    bytes_start = __decode_nnr_start_unit(nnr_gen, ndu_start, hls_stats )
    bytes_read = [bytes_start]
    
    while( bytes_read[0] < len(bitstream) ):
   
        reader = hls.BitReader(bitstream[bytes_read[0]:])
        
        bytes_ndu, mps, lps, tpl, model_info, approx_data, nnr_ndu_decoded, decoded_dc_tensorG = __decode_nnr_unit( reader, 
                                                                                                                    bitstream, 
                                                                                                                    bytes_read[0], 
                                                                                                                    mps, 
                                                                                                                    lps, 
                                                                                                                    tpl, 
                                                                                                                    model_info, 
                                                                                                                    approx_data, 
                                                                                                                    nnr_ndu_decoded, 
                                                                                                                    decoded_dc_tensorG, 
                                                                                                                    set_model_info, 
                                                                                                                    hls_stats
                                                                                                                )

        bytes_read[0] += bytes_ndu

    return approx_data
    
    

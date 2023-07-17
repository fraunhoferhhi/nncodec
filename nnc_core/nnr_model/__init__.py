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
import enum
from abc import ABC, abstractmethod
import nnc_core
import numpy as np
from nnc_core import hls, common
import pickle

class TopologyStorageFormat(enum.IntEnum):
    NNR_TPL_UNREC   = 0
    NNR_TPL_NNEF    = 1
    NNR_TPL_ONNX    = 2
    NNR_TPL_PYT     = 3
    NNR_TPL_TEF     = 4
    NNR_TPL_PRUN    = 5
    NNR_TPL_REFLIST = 6
    
class TopologyCompressionFormat(enum.IntEnum):
    NNR_PT_RAW = 0
    NNR_DFL    = 1

W_TYPES = [
    "weight"
]
O_TYPES = [
    "weight.ls",
    "bias",
    "bn.beta",
    "bn.gamma",
    "bn.mean",
    "bn.var",
    "unspecified"
]

class ModelExecute(ABC):
    def eval_model(self,
                   parameters,
                   verbose=False,
                   ):
        
        raise NotImplementedError("Function eval_model not yet implemented. This function is e.g. required for Inference-optimised quantization (IOQ). Either implement eval_model or deactivate IOQ (controlled by parameter ioq)!")


    def test_model(self,
                   parameters,
                   verbose=False,
                   ):
        
        raise NotImplementedError("Function test_model not yet implemented. This function is e.g. required for inference.")
    
    
    def tune_model(self,
                   parameters,
                   param_types,
                   lsa_flag,
                   ft_flag,
                   verbose=False,
                   ):
        
        raise NotImplementedError("Function tune_model not yet implemented. This function is required for fine tuning (FT) and local scaling adaptation (LSA). Either implement the function or deactivate fine tuning (controlled by parameter ft) and local scaling adaptation (controlled by parameter lsa)!")
    
    @abstractmethod
    def has_eval(self):
        return False
    
    @abstractmethod
    def has_test(self):
        return False
    
    @abstractmethod
    def has_tune_ft(self):
        return False
    
    @abstractmethod
    def has_tune_lsa(self):
        return False
    
    
def create_NNC_model_instance_from_file( model_path ):

    try:
        parameter_dict = pickle.load( model_path )
 
        if isinstance(parameter_dict, dict) and all( [isinstance(a, np.ndarray) for a in parameter_dict.values()] ) and (all([a.dtype==np.float32 for a in parameter_dict.values()]) or  all([a.dtype==np.int32 for a in parameter_dict.values()])):
            NNCModel = NNRModel( parameter_dict )
        else:
            raise SystemExit("Parameter dict must be a dict (key-value pairs). The keys shall be stings, specifying the tensor names. The values must be numpy arrays (ndarray) of type float32 or int32!")
    except:
        raise SystemExit("Can't read model: {}".format(model_path))

    return NNCModel, parameter_dict


def save_to_pickled_file( model_dict, model_path ):
    pickle.dump( model_dict, open(model_path, "wb") )

class NNRModel():

    def __init__(self, model_dict=None):
        
        if model_dict and isinstance(model_dict, dict):
            self.init_model_from_dict(model_dict)
        else:
            self.__model_info = None

        self.model = None


    def init_model_from_dict(self, model_dict):
        
        if isinstance(model_dict, dict):
            model_dict = model_dict
        else:
            raise SystemExit("model_dict must be of type dict")

        model_data = {'parameters': {}, 'reduction_method': 'baseline'}
        model_info = {'parameter_type': {}, 'parameter_dimensions': {}, 'parameter_index': {}, 'block_identifier': {}, 'original_size': {}, 'topology_storage_format' : None, 'topology_compression_format' : None}

        type_list_int = ['int8', 'int16', 'int32', 'uint8', 'uint16', 'uint32']
        type_list_1_bytes = ['int8', 'uint8']
        type_list_2_bytes = ['int16', 'uint16', 'float16']
        original_size = 0

        for i, module_name in enumerate(model_dict):
            if model_dict[module_name].dtype in type_list_1_bytes:
                original_size += model_dict[module_name].size
            elif model_dict[module_name].dtype in type_list_2_bytes:
                original_size += model_dict[module_name].size*2
            else:
                original_size += model_dict[module_name].size*4
            model_data['parameters'][module_name] = np.int32(model_dict[module_name]) if model_dict[module_name].dtype in type_list_int else model_dict[module_name]
            mdl_shape = model_data['parameters'][module_name].shape
            model_info['parameter_dimensions'][module_name] = mdl_shape
            if len(mdl_shape) == 0: #scalar
                model_data['parameters'][module_name] = np.array([np.float32(model_data['parameters'][module_name])])
                model_info['parameter_dimensions'][module_name] = np.array([0]).shape
            model_info['parameter_index'][module_name] = i

            dims = len(mdl_shape)

            if dims > 1:
                model_info['parameter_type'][module_name] = 'weight'
            else:
                model_info['parameter_type'][module_name] = 'unspecified'

        model_info['topology_storage_format'] = nnc_core.nnr_model.TopologyStorageFormat.NNR_TPL_UNREC
        model_info['topology_compression_format'] = nnc_core.nnr_model.TopologyCompressionFormat.NNR_PT_RAW

        model_info["original_size"] = original_size

        self.__model_info = model_info

        return model_data["parameters"]
    
    def guess_block_id_and_param_type(self, model_parameters):
        raise SystemExit("Block id and parameter type can not be guessed for generic model class. Try to provide a pytorch model, a tensorflow model or block_id_and_param_type (see description of compress_model)!")
                    
    @property
    def model_info(self):
        return self.__model_info


class NNRParamAccess():
    def __init__(self, model_info, param):
        self.__single_param = (model_info["parameter_type"].get(param), param, model_info["parameter_dimensions"].get(param))

    def param_generator(self, cpt_dict_dummy):
        yield self.__single_param

    @property
    def block_id(self):
        return None

    @property
    def param(self):
        return self.__single_param[1]
    

class NNRBlockAccess():
    def __init__(self, model_info, block_identifier):
        self.__block_identifier = block_identifier
        self.__model_info = model_info
        block_list = [
            x
            for x in model_info["block_identifier"]
            if model_info["block_identifier"][x] == block_identifier
        ] 
        self.__block_dict = { model_info["parameter_type"][x]: x for x in block_list }


    @property
    def block_id(self):
        return self.__block_identifier

    @property
    def w(self):
        for x in ["weight"]:
            if x in self.__block_dict:
                return self.__block_dict[x]

    @property
    def dc_g(self):
        return self.w + "_G"

    @property
    def dc_h(self):
        return self.w + "_H"

    @property
    def ls(self):
        return self.w + "_scaling"
            
    @property
    def bn_beta(self):
        return self.__block_dict.get("bn.beta", None)

    @property
    def bn_gamma(self):
        return self.__block_dict.get("bn.gamma", None)

    @property
    def bn_mean(self):
        return self.__block_dict.get("bn.mean", None)

    @property
    def bn_var(self):
        return self.__block_dict.get("bn.var", None)

    @property
    def bi(self):
        for x in ["bias"]:
            if x in self.__block_dict:
                return self.__block_dict[x]
        for x in ["weight"]:
            if x in self.__block_dict:
                return self.__block_dict[x] + ".bias"

    
    def param_generator(self, compressed_parameter_types_dict):
        compressed_parameter_types = compressed_parameter_types_dict[self.block_id]

        if compressed_parameter_types & hls.BlockParameterTypes.NNR_CPT_LS != 0: 
            yield "weight.ls", self.ls, [self.__model_info["parameter_dimensions"][self.w][0]]
        if compressed_parameter_types & hls.BlockParameterTypes.NNR_CPT_BI != 0:
            yield "bias", self.bi, [self.__model_info["parameter_dimensions"][self.w][0]]
        if compressed_parameter_types & hls.BlockParameterTypes.NNR_CPT_BN != 0:
            yield "bn.beta", self.bn_beta, self.__model_info["parameter_dimensions"][self.bn_beta]
            yield "bn.gamma", self.bn_gamma, self.__model_info["parameter_dimensions"][self.bn_gamma]
            yield "bn.mean", self.bn_mean, self.__model_info["parameter_dimensions"][self.bn_mean]
            yield "bn.var", self.bn_var, self.__model_info["parameter_dimensions"][self.bn_var]
        if compressed_parameter_types & hls.BlockParameterTypes.NNR_CPT_DC != 0:
            yield "weight", self.dc_g, self.__model_info["parameter_dimensions"][self.w]
            yield "weight", self.dc_h, self.__model_info["parameter_dimensions"][self.w]
        else:
            yield "weight", self.w, self.__model_info["parameter_dimensions"][self.w]
            
    def topology_elem_generator(self, compressed_parameter_types_dict):
        compressed_parameter_types = compressed_parameter_types_dict[self.block_id]
        if compressed_parameter_types & hls.BlockParameterTypes.NNR_CPT_DC != 0:
            yield self.dc_g
            yield self.dc_h
        else:
            yield self.w
        if compressed_parameter_types & hls.BlockParameterTypes.NNR_CPT_LS != 0: 
            yield  self.ls
        if compressed_parameter_types & hls.BlockParameterTypes.NNR_CPT_BN != 0:
            yield self.bn_beta
            yield self.bn_gamma
            yield self.bn_mean
            yield self.bn_var
        if compressed_parameter_types & hls.BlockParameterTypes.NNR_CPT_BI != 0:
            yield self.bi
        

class NNRModelAccess():
    def __init__(self, model_info):
        self.__model_info = model_info
        self.__block_list = []
        block_set_check = set( model_info["block_identifier"].values() )
        params_sorted = sorted(model_info["parameter_index"], key=model_info["parameter_index"].get)
        for param in params_sorted:
            if param in model_info["block_identifier"]:
                if model_info["parameter_type"][param] in ["weight"]:
                    self.__block_list.append( (model_info["block_identifier"][param], param) )
                    block_set_check.remove( model_info["block_identifier"][param] )
            else:
                self.__block_list.append( (None, param) )
        assert not block_set_check, "Unresolved block identifiers: {}".format( block_set_check )

    def blocks_and_params(self):
        for block_id, param in self.__block_list:
            if block_id is None:
                yield NNRParamAccess(self.__model_info, param)
            else:
                yield NNRBlockAccess(self.__model_info, block_id)

        

def set_block_id_and_param_type(model_info, block_id_and_param_type):
    assert "block_identifier" in block_id_and_param_type, "block_identifier not available!"
    assert "parameter_type" in block_id_and_param_type, "parameter_type not available!"
    
    model_info["block_identifier"] = {}
    
    block_id_values_list = list(block_id_and_param_type["block_identifier"].values())

    for param, pardIdx in model_info["parameter_index"].items():
        model_info["parameter_index"][param] = pardIdx
        if param in block_id_and_param_type["parameter_type"].keys():
            model_info["parameter_type"][param] = block_id_and_param_type["parameter_type"][param]
        if param in block_id_and_param_type["block_identifier"].keys() and block_id_and_param_type["block_identifier"][param] is not None and block_id_values_list.count(block_id_and_param_type["block_identifier"][param]) > 1 :
            model_info["block_identifier"][param] = block_id_and_param_type["block_identifier"][param]
            
            
def add_lsa_to_block_id_and_param_type( block_id_and_param_type, lsa_params ):
    for key in lsa_params.keys():
        if key not in block_id_and_param_type["block_identifier"]:
            block_id_and_param_type["block_identifier"][key] = block_id_and_param_type["block_identifier"].get(key.strip("_scaling"), None)
            block_id_and_param_type["parameter_type"][key] = "weight.ls"
            

def sanity_check_block_id_and_param_type(block_id_and_param_type, model_parameters=None):
    block_dict = dict()
    sanity_check_success = True
    for param, blkId in block_id_and_param_type["block_identifier"].items():
        if blkId != None:
            parT = block_id_and_param_type["parameter_type"][param]
            parShape = model_parameters[param].shape if model_parameters else None
            if model_parameters and parT != "weight" and len(parShape) != 1:
                sanity_check_success = False
                break
            if blkId not in block_dict.keys():
                block_dict[blkId] = []
            block_dict[blkId].append([param, parT, parShape])
    
    for bId, bList in block_dict.items():
        available_types = ["weight", "weight.ls", "bias", "bn.mean", "bn.var", "bn.gamma", "bn.beta"]
        lastShape = None
        for par, parT, parShape in bList:
            if parT not in available_types and parT != "unspecified":
                sanity_check_success = False
                break
            if parT != "unspecified":
                available_types.remove(parT) 
            if lastShape != None and lastShape[0] != parShape[0]:
                sanity_check_success = False
                break
            lastShape = parShape
        if "weight" in available_types:
            sanity_check_success = False
            break
    
    return sanity_check_success
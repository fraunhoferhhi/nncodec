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

import copy, logging
LOGGER = logging.getLogger()
import nnc_core
import h5py
import os
import tensorflow as tf

import copy, logging
import numpy as np
from framework.use_case_init import use_cases
from framework.applications.utils import evaluation
from collections import OrderedDict

def is_tef_model( model_object ):
    return isinstance( model_object, tf.Module ) 
    

def save_to_tensorflow_file( model_data, path ):
        h5_model = h5py.File(path, 'w')

        grp_names = []
        for module_name in model_data:
            splits = module_name.split('/')
            grp_name = module_name.split('/')[0]
            if splits[0] == splits[2]:
                grp_name = (splits[0] + '/' + splits[1])
            if grp_name not in grp_names:
                grp_names.append(grp_name)
            h5_model.create_dataset(module_name, data=model_data[module_name])
        h5_model.attrs['layer_names'] = grp_names

        for grp in h5_model:
            weight_attr = []
            if isinstance(h5_model[grp], h5py.Group) and grp in grp_names:
                weight_attr = [k[len(grp)+1:] for k, v in model_data.items()
                               if k.startswith(grp+'/')]
                h5_model[grp].attrs['weight_names'] = weight_attr
            elif isinstance(h5_model[grp], h5py.Group):
                for subgrp in h5_model[grp]:
                    if isinstance(h5_model[grp], h5py.Group) and (grp + '/' + subgrp) in grp_names:
                        weight_attr = [k[len(grp) + len(subgrp) + 2:] for k, v in model_data.items()
                                       if k.startswith(grp + '/' + subgrp + '/')]
                    h5_model[grp + '/' + subgrp].attrs['weight_names'] = weight_attr

def __initialize_data_functions(
                 handler=None,
                 dataset_path=None,
                 batch_size=None,
                 num_workers=None,
                 model_name='unspecified'
                ):

    if model_name==None:
        model_name='unspecified'

    if dataset_path:
        test_set, test_loader = handler.init_test_tef(
            dataset_path,
            batch_size,
            num_workers,
            model_name
        )

        val_set, val_loader = handler.init_validation_tef(
            dataset_path,
            batch_size,
            num_workers,
            model_name
        )

    else:
        test_set, test_loader = None, None
        val_set, val_loader   = None, None
    
    return test_set, test_loader, val_set, val_loader

def create_NNC_model_instance_from_file(
                 model_path,
                 dataset_path=None,
                 lr=1e-5,
                 batch_size=64,
                 num_workers=1,
                 model_struct=None,
                 model_name=None
                ):

    TEFModel = TensorFlowModel()
    model_parameters, loaded_model_struct = TEFModel.load_model(model_path)
    
    if model_struct == None and loaded_model_struct != None:
        model_struct = loaded_model_struct

    if dataset_path and model_struct:
        TEFModelExecuter = create_imagenet_model_executer(model_struct=model_struct,
                                                          dataset_path=dataset_path,
                                                          lr=lr,
                                                          batch_size=batch_size,
                                                          num_workers=num_workers,
                                                          model_name=model_name)

    else:
        TEFModelExecuter = None

    return TEFModel, TEFModelExecuter, model_parameters


def create_NNC_model_instance_from_object(
                 model_object,
                 dataset_path=None,
                 lr=1e-5,
                 batch_size=64,
                 num_workers=1,
                 model_struct=None,
                 model_name=None
                ):

    TEFModel = TensorFlowModel()
    model_parameters, loaded_model_struct = TEFModel.init_model_from_model_object(model_object)
    
    if model_struct == None and loaded_model_struct != None:
        model_struct = loaded_model_struct

    if dataset_path and model_struct:
        TEFModelExecuter = create_imagenet_model_executer(model_struct=model_struct,
                                                          dataset_path=dataset_path,
                                                          lr=lr,
                                                          batch_size=batch_size,
                                                          num_workers=num_workers,
                                                          model_name=model_name)

    else:
        TEFModelExecuter = None

    return TEFModel, TEFModelExecuter, model_parameters


def create_imagenet_model_executer( 
                            model_struct,
                            dataset_path,
                            lr=1e-5,
                            batch_size=64,
                            num_workers=1,
                            model_name=None,
                            ):
    
    assert model_struct != None, "model_struct must be specified in order to create a model_executer!"
    assert dataset_path != None, "dataset_path must be specified in order to create a model_executer!"
    
    handler = use_cases['NNR_TEF']

    test_set, test_loader, val_set, val_loader = __initialize_data_functions( handler=handler,
                                                                              dataset_path=dataset_path,
                                                                              batch_size=batch_size,
                                                                              num_workers=num_workers,
                                                                              model_name=model_name)

    assert (test_set!=None and test_loader!= None) or ( val_set!= None and val_loader!= None ), "Any of the pairs test_set/test_loader or val_set/val_loader must be specified in order to use data driven optimizations methods!"
    TEFModelExecuter = ImageNetTensorFlowModelExecuter( handler,
                                                        train_loader=None,
                                                        test_loader=test_loader,
                                                        test_set=test_set,
                                                        val_loader=val_loader,
                                                        val_set=val_set,
                                                        model_struct=model_struct)

    return TEFModelExecuter


def get_model_file_with_parameters( parameters, model_struct ):
    new_model_struct = copy.deepcopy(model_struct)
    
    save_to_tensorflow_file(parameters, './intermed_h5_model.h5')
    new_model_struct.load_weights('./intermed_h5_model.h5')
    os.remove('./intermed_h5_model.h5')

    return new_model_struct


class TensorFlowModel(nnc_core.nnr_model.NNRModel):

    def __init__(self, model_dict=None):
        
        if model_dict and isinstance(model_dict, dict):
            self.init_model_from_dict(model_dict)
        else:
            self.__model_info = None

        self.model = None

    def load_model( self, 
                    model_path
                  ):
        model_file = h5py.File(model_path, 'r')
        
        try:
            if isinstance(model_file, dict):
                model_parameter_dict = model_file
            elif isinstance(model_file, tf.Module):
                return self.init_model_from_model_object(model_file)
            else:
                if 'layer_names' in model_file.attrs:
                    module_names = [n for n in model_file.attrs['layer_names']]

                layer_names = []
                for mod_name in module_names:
                    layer = model_file[mod_name]
                    if 'weight_names' in layer.attrs:
                        weight_names = [mod_name+'/'+n for n in layer.attrs['weight_names']]
                        if weight_names:
                            layer_names += weight_names

                model_parameter_dict = {}
                for name in layer_names:
                    model_parameter_dict[name] = model_file[name]
        except:
            raise SystemExit("Can't read model: {}".format(model_path))

        return self.init_model_from_dict( model_parameter_dict ), None


    def init_model_from_model_object(   self,
                                        model_object,
                                    ):
        self.model = model_object

        h5_model_path = './temp.h5'
        model_object.save_weights(h5_model_path)
        model = h5py.File(h5_model_path, 'r')
        os.remove(h5_model_path)
        
        if 'layer_names' in model.attrs:
            module_names = [n for n in model.attrs['layer_names']]

        layer_names = []
        for mod_name in module_names:
            layer = model[mod_name]
            if 'weight_names' in layer.attrs:
                weight_names = [mod_name+'/'+n for n in layer.attrs['weight_names']]
                if weight_names:
                    layer_names += weight_names

        model_parameter_dict = {}
        for name in layer_names:
            model_parameter_dict[name] = model[name]

        return self.init_model_from_dict( model_parameter_dict ), model_object


    @property
    def model_info(self):
        return self.__model_info


    def init_model_from_dict(self, tf_dict):


        model_data = {'parameters': {}, 'reduction_method': 'baseline'}
        model_info = {'parameter_type': {}, 'parameter_dimensions': {}, 'parameter_index': {}, 'block_identifier': {}, 'original_size': {}, 'topology_storage_format' : None, 'topology_compression_format' : None}

        type_list_int = ['int8', 'int16', 'int32', 'uint8', 'uint16', 'uint32']
        type_list_1_bytes = ['int8', 'uint8']
        type_list_2_bytes = ['int16', 'uint16', 'float16']
        original_size = 0

        for i, module_name in enumerate(tf_dict):
            model_data['parameters'][module_name] = tf_dict[module_name][()]
            if  model_data['parameters'][module_name].dtype in type_list_1_bytes:
                original_size += model_data['parameters'][module_name].size
            elif  model_data['parameters'][module_name].dtype in type_list_2_bytes:
                original_size +=  model_data['parameters'][module_name].size*2
            else:
                original_size +=  model_data['parameters'][module_name].size*4
            model_data['parameters'][module_name] = np.int32(model_data['parameters'][module_name]) if model_data['parameters'][module_name].dtype in type_list_int else model_data['parameters'][module_name]
            mdl_shape = model_data['parameters'][module_name].shape
            model_info['parameter_dimensions'][module_name] = mdl_shape
            if len(mdl_shape) == 0: #scalar
                model_data['parameters'][module_name] = np.array([np.float32(tf_dict[module_name][()])])
                model_info['parameter_dimensions'][module_name] = np.array([0]).shape
            model_info['parameter_index'][module_name] = i

            dims = len(mdl_shape)

            quantize = True
            quantize_onedim = True

            if dims > 1 and 'kernel' in module_name:
                model_info['parameter_type'][module_name] = 'weight' if quantize else 'unspecified'
            elif dims > 1:
                model_info['parameter_type'][module_name] = 'weight' if quantize else 'unspecified'
            elif dims == 1:
                if 'bias' in module_name:
                    model_info['parameter_type'][module_name] = 'bias' if quantize_onedim else 'unspecified'
                elif 'beta' in module_name:
                    model_info['parameter_type'][module_name] = 'bn.beta' if quantize_onedim else 'unspecified'
                elif 'moving_mean' in module_name:
                    model_info['parameter_type'][module_name] = 'bn.mean' if quantize_onedim else 'unspecified'
                elif 'moving_variance' in module_name:
                    model_info['parameter_type'][module_name] = 'bn.var' if quantize_onedim else 'unspecified'
                elif 'gamma' in module_name:
                    model_info['parameter_type'][module_name] = 'bn.gamma' if quantize_onedim else 'unspecified'
                elif 'weight' in module_name:
                    model_info['parameter_type'][module_name] = 'weight' if quantize_onedim else 'unspecified'
                else:
                    model_info['parameter_type'][module_name] = 'unspecified'
            else:
                model_info['parameter_type'][module_name] = 'unspecified'

        model_info['topology_storage_format'] = nnc_core.nnr_model.TopologyStorageFormat.NNR_TPL_TEF
        model_info['topology_compression_format'] = nnc_core.nnr_model.TopologyCompressionFormat.NNR_PT_RAW

        model_info["original_size"] = original_size
        
        self.__model_info = model_info

        return model_data["parameters"]

    def save_state_dict(self, path, model_data):
        h5_model = h5py.File(path, 'w')

        grp_names = []
        for module_name in model_data:
            splits = module_name.split('/')
            grp_name = module_name.split('/')[0].encode('utf8')
            if splits[0] == splits[2]:
                grp_name = (splits[0] + '/' + splits[1]).encode('utf8')
            if grp_name not in grp_names:
                grp_names.append(grp_name)
            if model_data[module_name].size != 1:
                h5_model.create_dataset(module_name, data=model_data[module_name])
            else: #scalar
                h5_model.create_dataset(module_name, data=np.int64(model_data[module_name][0]))
        h5_model.attrs['layer_names'] = grp_names

        for grp in h5_model:
            weight_attr = []
            if isinstance(h5_model[grp], h5py.Group) and grp.encode('utf8') in grp_names:
                weight_attr = [k[len(grp)+1:].encode('utf8') for k, v in model_data.items()
                               if k.startswith(grp+'/')]
                h5_model[grp].attrs['weight_names'] = weight_attr
            elif isinstance(h5_model[grp], h5py.Group):
                for subgrp in h5_model[grp]:
                    if isinstance(h5_model[grp], h5py.Group) and (grp + '/' + subgrp).encode('utf8') in grp_names:
                        weight_attr = [k[len(grp) + len(subgrp) + 2:].encode('utf8') for k, v in model_data.items()
                                       if k.startswith(grp + '/' + subgrp + '/')]
                    h5_model[grp + '/' + subgrp].attrs['weight_names'] = weight_attr

    
    def get_model_struct(self):
        return self.model


    def guess_block_id_and_param_type(self, model_parameters):
        try:
            block_id_and_param_type = {"block_identifier" : {}, "parameter_type" : {}}
            block_dict = dict()
            blkNum = -1
            for param in model_parameters.keys():
                splitted_param = param.split("/")
                param_end = splitted_param[-1]
                base_block_id  = "/".join(splitted_param[0:-1])
                base_block_id  = "/".join(splitted_param[0:-1])+":" if len(splitted_param[0:-1]) != 0 else "genericBlk:"
                dims = len(model_parameters[param].shape)
                paramShape = model_parameters[param].shape
            

                if dims > 1 and ('kernel' in param_end or 'weight' in param_end):
                    paramType = 'weight'
                    blockId = base_block_id
                elif dims > 1:
                    paramType = 'weight'
                    blockId = base_block_id
                elif dims == 1:
                    if 'bias' in param_end or 'beta' in param_end: ##could also be bn.beta
                        paramType = 'bias'
                        blockId = base_block_id
                    elif 'running_mean' in param_end or 'moving_mean' in param_end:
                        paramType = 'bn.mean'
                        blockId = base_block_id
                    elif 'running_var' in param_end or 'moving_variance' in param_end:                        
                        paramType = 'bn.var'
                        blockId = base_block_id
                    elif 'weight_scaling' in param_end:
                        paramType = 'weight.ls'
                        blockId = base_block_id
                    elif 'gamma' in param_end:
                        paramType = 'bn.gamma'
                        blockId = base_block_id
                    elif 'weight' in param_end:
                        paramType = 'weight'
                        blockId = base_block_id
                    else:
                        paramType = 'unspecified'
                        blockId = None
                else:
                    paramType = 'unspecified'
                    blockId = None
                
                
                if blockId:
                    block_id = base_block_id + str(blkNum)
                    if block_id in block_dict.keys():
                        if any([a[1] == paramType for a in block_dict[block_id]]):
                            blkNum += 1
                        block_id = base_block_id + str(blkNum)
                        blockId = base_block_id + str(blkNum)
                    else:
                        blkNum += 1
                        block_id = base_block_id + str(blkNum)
                        blockId = base_block_id + str(blkNum)
                        
                    if block_id not in block_dict.keys():
                        block_dict[block_id] = []
                        
                    block_dict[block_id].append( [param, paramType, blockId, dims, paramShape] )
                else:
                    block_id_and_param_type["parameter_type"][param] = paramType
                    block_id_and_param_type["block_identifier"][param] = blockId
                
                
            weight_block_list = []
            bn_block_list     = []

            for block_list in block_dict.values():
                if any(["bn." in a[1] for a in block_list]):
                    for i, val in enumerate(block_list):
                        par, parT, blkId, dims, _ = val
                        if parT == 'weight' and dims == 1:
                            block_list[i][1] = "bn.gamma"
                        if parT == 'bias':
                            block_list[i][1] = "bn.beta"
                    bn_block_list.append( block_list )
                else:
                    weight_block_list.append(block_list)
                    
            weight_shape = None
            weight_blkId = None
            for weight_block in weight_block_list:
                weight_shape = None
                weight_blkId = None
                for par, parT, blkId, dims, paramSh in weight_block:
                    block_id_and_param_type["parameter_type"][par] = parT
                    block_id_and_param_type["block_identifier"][par] = blkId
                    if parT == 'weight':
                        weight_shape = paramSh
                        weight_blkId = blkId
            
                if len(bn_block_list) != 0 and any([dim == bn_block_list[0][0][4][0] for dim in weight_shape]):
                    bn_block = bn_block_list.pop(0)
                    for par, parT, _, _, _ in bn_block:
                        block_id_and_param_type["parameter_type"][par] = parT
                        block_id_and_param_type["block_identifier"][par] = weight_blkId
            
            assert len(bn_block_list) == 0, "Unhandled BN parameters!"
            
        except:
            print("INFO: Guessing of block_id_and_parameter_type failed! block_id_and_parameter_type has been set to 'None'!")
            block_id_and_param_type = None

        return block_id_and_param_type



class ImageNetTensorFlowModelExecuter( nnc_core.nnr_model.ModelExecute ):

    def __init__(self,
                 handler,
                 train_loader,
                 test_loader,
                 test_set,
                 val_loader,
                 val_set,
                 model_struct):
        
        self.handle = handler
        if test_set:
            self.test_set = test_set
            self.test_loader = test_loader
        if val_set:
            self.val_set = val_set
            self.val_loader = val_loader
        if train_loader:
            self.train_loader = train_loader

        if model_struct:
            self.original_model = model_struct
            self.model = model_struct
        else:
            self.original_model = None
            self.model = None

    
    def save_state_dict(self, path, model_data):
        h5_model = h5py.File(path, 'w')

        grp_names = []
        for module_name in model_data:
            splits = module_name.split('/')
            grp_name = module_name.split('/')[0].encode('utf8')
            if splits[0] == splits[2]:
                grp_name = (splits[0] + '/' + splits[1]).encode('utf8')
            if grp_name not in grp_names:
                grp_names.append(grp_name)
            if model_data[module_name].size != 1:
                h5_model.create_dataset(module_name, data=model_data[module_name])
            else: #scalar
                h5_model.create_dataset(module_name, data=np.int64(model_data[module_name][0]))
        h5_model.attrs['layer_names'] = grp_names

        for grp in h5_model:
            weight_attr = []
            if isinstance(h5_model[grp], h5py.Group) and grp.encode('utf8') in grp_names:
                weight_attr = [k[len(grp)+1:].encode('utf8') for k, v in model_data.items()
                               if k.startswith(grp+'/')]
                h5_model[grp].attrs['weight_names'] = weight_attr
            elif isinstance(h5_model[grp], h5py.Group):
                for subgrp in h5_model[grp]:
                    if isinstance(h5_model[grp], h5py.Group) and (grp + '/' + subgrp).encode('utf8') in grp_names:
                        weight_attr = [k[len(grp) + len(subgrp) + 2:].encode('utf8') for k, v in model_data.items()
                                       if k.startswith(grp + '/' + subgrp + '/')]
                    h5_model[grp + '/' + subgrp].attrs['weight_names'] = weight_attr

    def test_model(
            self,
            parameters,
            verbose=False
    ):
            tf.config.optimizer.set_jit(True)

            Model = self.model
            self.save_state_dict('./intermed_h5_model.h5', parameters)
            Model.load_weights('./intermed_h5_model.h5')
            os.remove('./intermed_h5_model.h5')

            acc = self.handle.evaluate( 
            Model,
            self.test_loader,
            self.test_set,
            num_workers=8,
            verbose=verbose
        )
            del Model   
            return acc

    def eval_model(
            self,
            parameters,
            verbose=False,
    ):

            tf.config.optimizer.set_jit(True)

            Model = self.model
            self.save_state_dict('./intermed_h5_model.h5', parameters)
            Model.load_weights('./intermed_h5_model.h5')
            os.remove('./intermed_h5_model.h5')

            acc = self.handle.evaluate( 
            Model,
            self.val_loader,
            self.val_set,
            num_workers=8,
            verbose=verbose
        )
            del Model   
            return acc

    def has_eval(self):
        return True
    
    def has_test(self):
        return True
    
    def has_tune_ft(self):
        return False
    
    def has_tune_lsa(self):
        return False

        
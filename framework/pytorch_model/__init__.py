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
import numpy as np
LOGGER = logging.getLogger()
import nnc_core
from framework.use_case_init import use_cases
from framework.applications.utils import evaluation, transforms
import torch
from collections import OrderedDict


def is_pyt_model( model_object ):
    return isinstance( model_object, torch.nn.Module ) 

def __initialize_data_functions(
                 handler=None,
                 dataset_path=None,
                 batch_size=None,
                 num_workers=None,
                ):

    if dataset_path: 
        test_set, test_loader = handler.init_test(
            dataset_path,
            batch_size,
            num_workers
        )

        val_set, val_loader = handler.init_validation(
            dataset_path,
            batch_size,
            num_workers
        )

        train_loader = handler.init_training(
            dataset_path,
            batch_size,
            num_workers
        )
    else:
        test_set, test_loader = None, None
        val_set, val_loader   = None, None
        train_loader = None
    
    return test_set, test_loader, val_set, val_loader, train_loader

def create_NNC_model_instance_from_file(
                 model_path,
                 dataset_path=None,
                 lr=1e-4,
                 epochs=30,
                 max_batches=None,
                 batch_size=64,
                 num_workers=1,
                 model_struct=None,
                 lsa=False
                ):

    PYTModel = PytorchModel()
    model_parameters, loaded_model_struct = PYTModel.load_model(model_path)
    if model_struct == None and loaded_model_struct != None:
        model_struct = loaded_model_struct

    if dataset_path and model_struct:
        PYTModelExecuter = create_imagenet_model_executer(model_struct=model_struct,
                                                          dataset_path=dataset_path,
                                                          lr=lr,
                                                          epochs=epochs,
                                                          max_batches=max_batches,
                                                          batch_size=batch_size,
                                                          num_workers=num_workers,
                                                          lsa=lsa
                                                          )
        if lsa:
            model_parameters = PYTModel.init_model_from_dict(PYTModelExecuter.model.state_dict())
    else:
        PYTModelExecuter = None

    return PYTModel, PYTModelExecuter, model_parameters


def create_NNC_model_instance_from_object(
                 model_object,
                 dataset_path=None,
                 lr=1e-4,
                 epochs=30,
                 max_batches=None,
                 batch_size=64,
                 num_workers=1,
                 model_struct=None,
                 lsa=False
                ):

    PYTModel = PytorchModel()
    model_parameters, loaded_model_struct = PYTModel.init_model_from_model_object(model_object)
    if model_struct == None and loaded_model_struct != None:
        model_struct = loaded_model_struct

    if dataset_path and model_struct:
        PYTModelExecuter = create_imagenet_model_executer(model_struct=model_struct,
                                                          dataset_path=dataset_path,
                                                          lr=lr,
                                                          epochs=epochs,
                                                          max_batches=max_batches,
                                                          batch_size=batch_size,
                                                          num_workers=num_workers,
                                                          lsa=lsa
                                                          )
        if lsa:
            model_parameters = PYTModel.init_model_from_dict(PYTModelExecuter.model.state_dict())
    else:
        PYTModelExecuter = None

    return PYTModel, PYTModelExecuter, model_parameters


def create_imagenet_model_executer( 
                            model_struct,
                            dataset_path,
                            lr=1e-4,
                            epochs=30,
                            max_batches=None,
                            batch_size=64,
                            num_workers=1,
                            lsa=False,
                            ):
    
    assert model_struct != None, "model_struct must be specified in order to create a model_executer!"
    assert dataset_path != None, "dataset_path must be specified in order to create a model_executer!"
    
    handler = use_cases['NNR_PYT']

    test_set, test_loader, val_set, val_loader, train_loader = __initialize_data_functions( handler=handler,
                                                                              dataset_path=dataset_path,
                                                                              batch_size=batch_size,
                                                                              num_workers=num_workers)

    assert (test_set!=None and test_loader!= None) or ( val_set!= None and val_loader!= None ), "Any of the pairs test_set/test_loader or val_set/val_loader must be specified in order to use data driven optimizations methods!"
    PYTModelExecuter = ImageNetPytorchModelExecuter(handler,
                                                    train_loader=train_loader,
                                                    test_loader=test_loader,
                                                    test_set=test_set,
                                                    val_loader=val_loader,
                                                    val_set=val_set,
                                                    model_struct=model_struct,
                                                    lsa=lsa,
                                                    lr=lr,
                                                    epochs=epochs,
                                                    max_batches=max_batches)

    PYTModelExecuter.initialize_optimizer(lr=lr)

    return PYTModelExecuter


def save_to_pytorch_file( model_data, path ):
    model_dict = OrderedDict()
    for module_name in model_data:
        model_dict[module_name] = torch.tensor(model_data[module_name])
    torch.save(model_dict, path)


def get_model_file_with_parameters( parameters, model_struct ):

    new_model_struct = copy.deepcopy(model_struct)

    state_dict = OrderedDict()
    for param in parameters.keys():
        state_dict[param] = torch.tensor( parameters[param] )
        assert param in new_model_struct.state_dict(), "The provided model_strcut does not fit the parameter state dict decoded from the bitstream! Parameter '{}' not found in model_struct state dict!".format(param)

    new_model_struct.load_state_dict(state_dict)

    return new_model_struct


class PytorchModel(nnc_core.nnr_model.NNRModel):
    
    def __init__(self, model_dict=None):
        
        if model_dict and isinstance(model_dict, dict):
            self.init_model_from_dict(model_dict)
        else:
            self.__model_info = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_model(self, 
                   model_path
                  ):
        
        model_file = torch.load(model_path, map_location=self.device) ##loads the state_dict
        model_struct = None
        
        try:
            model_parameter_dict = None

            # model state_dict
            if isinstance(model_file, OrderedDict):
                model_parameter_dict = model_file

            # checkpoint including state_dict
            elif isinstance(model_file, dict):
                for key in model_file.keys():
                    if isinstance(model_file[key], OrderedDict):
                        model_parameter_dict = model_file[key]
                        print("Loaded weights from state_dict '{}' from checkpoint elements {}".format(key,
                                                                                                       model_file.keys()))
                        break
                if not model_parameter_dict:
                    assert 0, "Checkpoint does not include a state_dict in {}".format(model_file.keys())

            # whole model (in general not recommended)
            elif isinstance(model_file, torch.nn.Module):
                model_parameter_dict = model_file.state_dict()
                model_struct = model_file

            # multi-GPU parallel trained models (torch.nn.DataParallel)
            if all(i[:7] == 'module.' for i in model_parameter_dict.keys()):
                print("Removing 'module.' prefixes from state_dict keys resulting from saving torch.nn.DataParallel "
                      "models not in the recommended way, that is torch.save(model.module.state_dict()")
                new_state_dict = OrderedDict()
                for n, t in model_parameter_dict.items():
                    name = n[7:]  # remove `module.`
                    new_state_dict[name] = t
                model_parameter_dict = new_state_dict

        except:
            raise SystemExit("Can't read model: {}".format(model_path))
        
        return self.init_model_from_dict(model_parameter_dict), model_struct ##intializes the model and the state dict and returns the PYTModel instance and parameters dict


    def init_model_from_model_object(   self,
                                        model_object,
                                    ):

        return self.init_model_from_dict(model_object.state_dict()), model_object


    @property
    def model_info(self):
        return self.__model_info

    def init_model_from_dict(self, pt_dict):

        if isinstance(pt_dict, dict):
            model_dict = pt_dict
        elif isinstance(pt_dict.state_dict(), dict):
            model_dict = pt_dict.state_dict()

        model_data = {'parameters': {}, 'reduction_method': 'baseline'}
        model_info = {'parameter_type': {}, 'parameter_dimensions': {}, 'parameter_index': {}, 'block_identifier': {}, 'original_size': {},
                      'topology_storage_format': nnc_core.nnr_model.TopologyStorageFormat.NNR_TPL_PYT,
                      'topology_compression_format': nnc_core.nnr_model.TopologyCompressionFormat.NNR_PT_RAW}

        # metadata only needed for MNASNet from PYT model zoo... further work: include into bitstream
        # self.metadata = getattr(model_dict, '_metadata', None)

        type_list_int = ['int8', 'int16', 'int32', 'uint8', 'uint16', 'uint32']
        type_list_1_bytes = ['int8', 'uint8']
        type_list_2_bytes = ['int16', 'uint16', 'float16']
        original_size = 0

        for i, module_name in enumerate(model_dict):
            if '.num_batches_tracked' in module_name:
                continue
            if model_dict[module_name].data.cpu().detach().numpy().dtype in type_list_1_bytes:
                original_size += model_dict[module_name].numel()
            elif model_dict[module_name].data.cpu().detach().numpy().dtype in type_list_2_bytes:
                original_size += model_dict[module_name].numel()*2
            else:
                original_size += model_dict[module_name].numel()*4
            model_data['parameters'][module_name] = np.int32(model_dict[module_name].data.cpu().detach().numpy()) if model_dict[module_name].data.cpu().detach().numpy().dtype in type_list_int else model_dict[module_name].data.cpu().detach().numpy()
            if '.weight_scaling' in module_name:
                model_data['parameters'][module_name] = model_data['parameters'][module_name].flatten()
            mdl_shape = model_data['parameters'][module_name].shape
            model_info['parameter_dimensions'][module_name] = mdl_shape
            if len(mdl_shape) == 0:  # scalar
                model_data['parameters'][module_name] = np.array([np.float32(model_data['parameters'][module_name])])
                model_info['parameter_dimensions'][module_name] = np.array([0]).shape
            model_info['parameter_index'][module_name] = i

            dims = len(mdl_shape)

            if dims > 1 and '.weight' in module_name:
                model_info['parameter_type'][module_name] = 'weight'
            elif dims > 1:
                model_info['parameter_type'][module_name] = 'weight'
            elif dims == 1:
                if '.bias' in module_name:
                    model_info['parameter_type'][module_name] = 'bias' 
                elif '.running_mean' in module_name:
                    model_info['parameter_type'][module_name] = 'bn.mean'
                elif '.running_var' in module_name:
                    model_info['parameter_type'][module_name] = 'bn.var'
                elif '.weight_scaling' in module_name:
                    model_info['parameter_type'][module_name] = 'weight.ls'
                elif 'gamma' in module_name:
                    model_info['parameter_type'][module_name] = 'bn.gamma'
                elif '.weight' in module_name:
                    model_info['parameter_type'][module_name] = "weight"
                else:
                    model_info['parameter_type'][module_name] = 'unspecified'
            else:
                model_info['parameter_type'][module_name] = 'unspecified'
            
        model_info["original_size"] = original_size

        self.__model_info = model_info

        return model_data["parameters"]

    def save_state_dict(self, path, model_data):
        model_dict = OrderedDict()
        for module_name in model_data:
            model_dict[module_name] = torch.tensor(model_data[module_name])
            if model_data[module_name].size == 1:
                model_dict[module_name] = torch.tensor(np.int64(model_data[module_name][0]))
        torch.save(model_dict, path)
    
    def guess_block_id_and_param_type(self, model_parameters):
        
        try:
            block_id_and_param_type = {"block_identifier" : {}, "parameter_type" : {}}
            block_dict = dict()
            blkNum = -1
            for param in model_parameters.keys():
                dims = len(model_parameters[param].shape)
                paramShape = model_parameters[param].shape
                splitted_param = param.split(".")
                param_end = splitted_param[-1]
                base_block_id  = ".".join(splitted_param[0:-1]+[""]) if len(splitted_param[0:-1]) != 0 else "genericBlk."


                if dims > 1 and ('kernel' in param_end or 'weight' in param_end):
                    paramType = 'weight'
                    blockId = base_block_id
                elif dims > 1:
                    paramType = 'weight'
                    blockId = base_block_id
                elif dims == 1:
                    if 'bias' in param_end or 'beta' in param_end:
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
            
            assert len(bn_block_list) == 0
                                
        except:
            print("INFO: Guessing of block_id_and_parameter_type failed! block_id_and_parameter_type has been set to 'None'!")
            block_id_and_param_type = None
                
        return block_id_and_param_type


class ImageNetPytorchModelExecuter( nnc_core.nnr_model.ModelExecute):

    def __init__(self, 
                 handler,
                 train_loader,
                 test_loader,
                 test_set,
                 val_loader,
                 val_set,
                 model_struct,
                 lsa,
                 max_batches=None,
                 epochs=5,
                 lr=1e-4,
                 ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        torch.manual_seed(451)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.learning_rate = lr
        self.epochs = epochs
        self.max_batches = max_batches
        
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
            self.original_model = copy.deepcopy(model_struct)
            if lsa:
                lsa_gen = transforms.LSA(model_struct)
                model_struct = lsa_gen.add_lsa_params()
            self.model = model_struct
        else:
            self.original_model = None
            self.model = None

    def test_model(self,
                   parameters,
                   verbose=False
                   ):

        torch.set_num_threads(1)
        Model = copy.deepcopy(self.model)

        base_model_arch = Model.state_dict()
        model_dict = OrderedDict()
        for module_name in base_model_arch:
            if module_name in parameters:
                model_dict[module_name] = parameters[module_name] if torch.is_tensor(parameters[module_name]) else \
                    torch.tensor(parameters[module_name])
                if "weight_scaling" in module_name:
                   model_dict[module_name] = model_dict[module_name].reshape(base_model_arch[module_name].shape)

        # metadata only needed for MNASNet from PYT model zoo... further work: include into bitstream
        # if self.metadata is not None:
        #     model_dict._metadata = self.metadata

        for param in Model.state_dict().keys():
            if "num_batches_tracked" in param:
                continue
            assert param in parameters.keys(), "The provided model_strcut does not fit the parameter state dict decoded from the bitstream! Parameter '{}' not found in parameter dict!".format(param)
        
        Model.load_state_dict(model_dict)

        acc = self.handle.evaluate(
            Model,
            self.handle.criterion,
            self.test_loader,
            self.test_set,
            device=self.device,
            verbose=verbose
        )
        del Model
        return acc

    def eval_model(self,
                   parameters,
                   verbose=False
                   ):

        torch.set_num_threads(1)

        Model = copy.deepcopy(self.model)

        base_model_arch = Model.state_dict()
        model_dict = OrderedDict()
        for module_name in base_model_arch:
            if module_name in parameters:
                model_dict[module_name] = parameters[module_name] if torch.is_tensor(parameters[module_name]) else \
                    torch.tensor(parameters[module_name])
                if "weight_scaling" in module_name:
                   model_dict[module_name] = model_dict[module_name].reshape(base_model_arch[module_name].shape)

        # metadata only needed for MNASNet from PYT model zoo... further work: include into bitstream
        # if self.metadata is not None:
        #     model_dict._metadata = self.metadata

        Model.load_state_dict(model_dict)

        acc = self.handle.evaluate(
            Model,
            self.handle.criterion,
            self.val_loader,
            self.val_set,
            device=self.device,
            verbose=verbose
        )

        del Model
        return acc

    def tune_model(
            self,
            parameters,
            param_types,
            lsa_flag=False,
            ft_flag=False,
            verbose=False,     
    ):
        torch.set_num_threads(1)
        verbose = 1 if (verbose & 1) else 0

        base_model_arch = self.model.state_dict()
        model_dict = OrderedDict()
        for module_name in base_model_arch:
            if module_name in parameters:
                model_dict[module_name] = parameters[module_name] if torch.is_tensor(parameters[module_name]) else \
                    torch.tensor(parameters[module_name])
                if "weight_scaling" in module_name:
                   model_dict[module_name] = model_dict[module_name].reshape(base_model_arch[module_name].shape)
        self.model.load_state_dict(model_dict)

        for param in parameters:
            parameters[param] = copy.deepcopy(self.model.state_dict()[param])

        tuning_params = []
        for name, param in self.model.named_parameters():
            if lsa_flag and ft_flag and param_types[name] in nnc_core.nnr_model.O_TYPES:
                param.requires_grad = True
                tuning_params.append(param)
            elif lsa_flag and param_types[name] == 'weight.ls':
                param.requires_grad = True
                tuning_params.append(param)
            elif ft_flag and param_types[name] != 'weight.ls' and param_types[name] in nnc_core.nnr_model.O_TYPES:
                param.requires_grad = True
                tuning_params.append(param)
            else:
                param.requires_grad = False

        self.tuning_optimizer = torch.optim.Adam(tuning_params, lr=self.learning_rate)

        perf = self.eval_model(parameters, verbose=verbose)
        best_loss, best_params = perf[2], copy.deepcopy(parameters)
        if verbose:
            print(f'Validation accuracy (loss) before LSA and/or Fine Tuning: {perf[0]} ({perf[2]})')
            print(f'Test performance (top1, top5, loss) before LSA and/or Fine Tuning: '
                  f'{self.test_model(parameters, verbose=verbose)}')
        for e in range(self.epochs):
            train_acc, loss = self.handle.train(
                self.model,
                self.tuning_optimizer,
                self.handle.criterion,
                self.train_loader,
                device=self.device,
                verbose=verbose,
                freeze_batch_norm=True if lsa_flag and not ft_flag else False,
                max_batches=self.max_batches if self.max_batches else None
            )
            print(f'Epoch {e+1}: Train accuracy: {train_acc}, Loss: {loss}')
            for param in parameters:
                parameters[param] = copy.deepcopy(self.model.state_dict()[param])
            perf = self.eval_model(parameters, verbose=verbose)
            if perf[2] < best_loss and best_loss - perf[2] > 1e-3:
                best_loss = perf[2]
                best_params = copy.deepcopy(parameters)
            else:
                if verbose:
                    print(f'Early Stopping due to model convergence or overfitting')
                    print(f'Epoch {e + 1}: Validation accuracy (loss) after Model Tuning: {perf[0]} ({perf[2]})')
                break
            if verbose:
                if lsa_flag and not ft_flag:
                    print(f'Epoch {e+1}: Validation accuracy (loss) after Model Tuning: {perf[0]} ({perf[2]})')
        if verbose:
            print(f'Test performance (top1, top5, loss) after LSA and/or Fine Tuning: '
                  f'{self.test_model(parameters, verbose=verbose)}')

        lsa_params, ft_params = {}, {}
        for name in best_params:
            if lsa_flag and param_types[name] == 'weight.ls':
                lsa_params[name] = best_params[name].cpu().numpy().flatten()
            elif ft_flag and param_types[name] != 'weight.ls' and param_types[name] in nnc_core.nnr_model.O_TYPES:
                ft_params[name] = best_params[name].cpu().numpy()
        return (lsa_params, ft_params)


    def initialize_optimizer(self,
                             lr=1e-5,
                             mdl_params=None
                             ):
        if mdl_params:
            mdl_params_copy = copy.deepcopy(mdl_params)
            Model = self.model
            base_model_arch = Model.state_dict()
            model_dict = OrderedDict()
            for module_name in base_model_arch:
                if module_name in mdl_params_copy:
                    model_dict[module_name] = torch.tensor(mdl_params_copy[module_name])
                else:
                    model_dict[module_name] = base_model_arch[module_name]
            Model.load_state_dict(model_dict)

        if hasattr(self, "model"):
            params = [param for name, param in self.model.named_parameters()
                        if not '.weight_scaling' in name]
            self.optimizer = torch.optim.Adam(params, lr=lr)


    def has_eval(self):
        return True
    
    def has_test(self):
        return True
    
    def has_tune_ft(self):
        return True
    
    def has_tune_lsa(self):
        return True
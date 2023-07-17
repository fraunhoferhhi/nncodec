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
    
import sys

assert sys.version_info >= (3, 6)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import copy
from timeit import default_timer as timer

import nnc_core
from nnc_core import nnr_model

from framework import tensorflow_model
from framework import pytorch_model

#import torch

def __print_output_line( outputString, verbose=True ):
    if verbose:
        sys.stdout.write(outputString)
        sys.stdout.flush()
        
def guess_block_id_and_param_type(model_struct, add_lsa_params=False):
    if tensorflow_model.is_tef_model(model_struct):
        nnc_mdl, _, model_parameters = tensorflow_model.create_NNC_model_instance_from_object(
                 model_struct,
                )
        block_id_and_param_type = nnc_mdl.guess_block_id_and_param_type(model_parameters)
    elif pytorch_model.is_pyt_model(model_struct):
        nnc_mdl, _, model_parameters = pytorch_model.create_NNC_model_instance_from_object(
                 model_struct,
                )
        block_id_and_param_type = nnc_mdl.guess_block_id_and_param_type(model_parameters)
    else:
        print("INFO: guess_block_id_and_param_type is only applicable to Pytorch and Tensorflow models! block_id_and_param_type has been set to 'None'")
        block_id_and_param_type=None
        
    if block_id_and_param_type and add_lsa_params:
        for param, parType in block_id_and_param_type["parameter_type"].items():
            if parType == "weight":
                lsa_param = param + "_scaling"
                block_id_and_param_type["parameter_type"][lsa_param] = "weight.ls"
                block_id_and_param_type["block_identifier"][lsa_param] = block_id_and_param_type["block_identifier"][param]
    
    if block_id_and_param_type:    
        blkIdParamTypeOk = nnc_core.nnr_model.sanity_check_block_id_and_param_type( block_id_and_param_type, model_parameters )
        if blkIdParamTypeOk == False:
            print("INFO: Sanity check for block_id_and_param_type failed! block_id_and_param_type has been set to 'None'!")
            block_id_and_param_type = None
    
    return block_id_and_param_type
        

def compress_model( model_path_or_object,
                    bitstream_path="./bitstream.nnc",
                    qp=-38,
                    qp_density=2,
                    nonweight_qp=-75,
                    qp_per_tensor=None,
                    use_dq=True,
                    codebook_mode=0,
                    scan_order=0,
                    lambda_scale=0,
                    param_opt=True,
                    cabac_unary_length_minus1=10,
                    opt_qp=False,
                    ioq=False,
                    bnf=False,
                    lsa=False,
                    fine_tune=False,
                    block_id_and_param_type=None,
                    model_name=None,
                    model_executer=None,
                    model_struct=None,
                    dataset_path=None,
                    learning_rate=1e-4,
                    batch_size=64,
                    epochs=30,
                    max_batches=600,
                    num_workers=8,
                    return_model_data=False,
                    verbose=True,
                    return_bitstream=False,
                   ):

    is_pyt_model = False
    is_tef_model = False
    dataset_path = None if dataset_path is None else os.path.expanduser(dataset_path)
        
    if tensorflow_model.is_tef_model(model_path_or_object): 
        if bnf:
            print("WARNING: Batch-norm folding (BNF) requires the tensors to be shaped such that the first dimensions corresponds to the number of output channels, which is usually not the case for TensorFlow. For further details refer to the Wiki!")
        if lsa:
            print("INFO: LSA not yet supported for TensorFlow models. 'lsa' has been set to false!")
            lsa = False
        is_tef_model = True
        if model_executer:
                nnc_mdl, _, model_parameters = tensorflow_model.create_NNC_model_instance_from_object(
                 model_path_or_object,
                )
        else:
            nnc_mdl, nnc_mdl_executer, model_parameters = tensorflow_model.create_NNC_model_instance_from_object(
                model_path_or_object,
                dataset_path=dataset_path,
                lr=learning_rate,
                batch_size=batch_size,
                num_workers=num_workers,
                model_struct=model_struct,
                model_name=model_name
                )
    elif pytorch_model.is_pyt_model(model_path_or_object):
        is_pyt_model = True
        if model_executer:
                nnc_mdl, _, model_parameters = pytorch_model.create_NNC_model_instance_from_object(
                 model_path_or_object,
                )
        else:    
            nnc_mdl, nnc_mdl_executer, model_parameters = pytorch_model.create_NNC_model_instance_from_object(
                model_path_or_object,
                dataset_path=dataset_path,
                lr=learning_rate,
                batch_size=batch_size,
                num_workers=num_workers,
                model_struct=model_struct,
                lsa=lsa,
                epochs=epochs,
                max_batches=max_batches,
                )
    elif os.path.exists( os.path.expanduser(model_path_or_object)):
        model_path_or_object = os.path.expanduser(model_path_or_object)
        if model_path_or_object.endswith(".h5") or model_path_or_object.endswith(".hdf5") or model_path_or_object.endswith(".tf"):
            if bnf:
                print("WARNING: Batch-norm folding (BNF) requires the tensors to be shaped such that the first dimensions corresponds to the number of output channels, which is usually not the case for TensorFlow. For further details refer to the Wiki!")
            if lsa:
                print("INFO: LSA not yet supported for TensorFlow models. 'lsa' has been set to false!")
                lsa = False
            is_tef_model = True
            if model_executer:
                nnc_mdl, _, model_parameters = tensorflow_model.create_NNC_model_instance_from_file(
                 model_path_or_object,
                )
            else:
                nnc_mdl, nnc_mdl_executer, model_parameters = tensorflow_model.create_NNC_model_instance_from_file(
                    model_path_or_object,
                    dataset_path=dataset_path,
                    lr=learning_rate,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    model_struct=model_struct,
                    model_name=model_name
                    )

        elif model_path_or_object.endswith(".pt") or model_path_or_object.endswith(".pth"):
            is_pyt_model = True
            if model_executer:
                nnc_mdl, _, model_parameters = pytorch_model.create_NNC_model_instance_from_file(
                 model_path_or_object,
                )
            else:    
                nnc_mdl, nnc_mdl_executer, model_parameters = pytorch_model.create_NNC_model_instance_from_file(
                    model_path_or_object,
                    dataset_path=dataset_path,
                    lr=learning_rate,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    model_struct=model_struct,
                    lsa=lsa,
                    epochs=epochs,
                    max_batches=max_batches,
                    )

        else:
            nnc_mdl, model_parameters = nnr_model.create_NNC_model_instance_from_file( model_path_or_object )
            nnc_mdl_executer = None

    else:
        raise SystemExit("Can't find path or object {}".format(model_path_or_object))

    if model_executer:
        nnc_mdl_executer = model_executer    
    
    if block_id_and_param_type is None and (bnf or lsa) and (is_pyt_model or is_tef_model):
        block_id_and_param_type = nnc_mdl.guess_block_id_and_param_type(model_parameters)
        blkIdParamTypeOk = nnc_core.nnr_model.sanity_check_block_id_and_param_type( block_id_and_param_type, model_parameters )
        if blkIdParamTypeOk == False:
            print("INFO: Sanity check for block_id_and_param_type failed! block_id_and_param_type has been set to 'None', and the flags 'lsa' and 'bnf' have been set to 'False'!")
            block_id_and_param_type = None
            lsa = False
            bnf = False
            if model_executer:
                model_executer.model = model_executer.original_model
                del model_executer.original_model
            for key in model_parameters.keys():
                if "weight_scaling" in key:
                    del model_parameters[key]
            

    bitstream = compress(   model_parameters,
                            bitstream_path=bitstream_path,
                            qp=qp,
                            qp_density=qp_density,
                            nonweight_qp=nonweight_qp,
                            qp_per_tensor=qp_per_tensor,
                            use_dq=use_dq,
                            codebook_mode=codebook_mode,
                            scan_order=scan_order,
                            lambda_scale=lambda_scale,
                            param_opt=param_opt,
                            cabac_unary_length_minus1=cabac_unary_length_minus1,
                            opt_qp=opt_qp,
                            ioq=ioq,
                            bnf=bnf,
                            lsa=lsa,
                            fine_tune=fine_tune,
                            block_id_and_param_type=block_id_and_param_type,
                            model=nnc_mdl,
                            model_executer=nnc_mdl_executer,
                            verbose=verbose,
                            return_bitstream=return_bitstream,
                            )

    if return_model_data==True:
        if return_bitstream:
            return bitstream, block_id_and_param_type
        else:
            return block_id_and_param_type
    elif return_bitstream:
        return bitstream


def compress( 
    parameter_dict,
    bitstream_path="./bitstream.nnc",
    qp=-38,
    qp_density=2,
    nonweight_qp=-75,
    qp_per_tensor=None,
    use_dq=True,
    codebook_mode=0,
    scan_order=0,
    lambda_scale=0,
    param_opt=True,
    cabac_unary_length_minus1=10,
    opt_qp=False,
    ioq=False,
    bnf=False,
    lsa=False,
    fine_tune=False,
    block_id_and_param_type=None,
    model=None,
    model_executer=None,
    verbose=True,
    return_bitstream=False,
    ):

    try:
        start = timer()
        start_overall = start
        __print_output_line("INITIALIZE APPROXIMATOR AND ENCODER...", verbose=verbose)
        if isinstance(parameter_dict, dict) and all( [isinstance(a, np.ndarray) for a in parameter_dict.values()] ) and (all([ (a.dtype==np.float32 or a.dtype==np.int32) for a in parameter_dict.values()])):
            model_parameters = parameter_dict
            
            if isinstance(model, nnc_core.nnr_model.NNRModel):
                nnc_mdl = model
            else:
                nnc_mdl = nnc_core.nnr_model.NNRModel(parameter_dict)

            if model_executer is not None:
                assert isinstance( model_executer, nnc_core.nnr_model.ModelExecute ), "model_executer must be of type ModelExecute!"
        else:
            raise SystemExit("Parameter dict must be a dict (key-value pairs). The keys shall be stings, specifying the tensor names. The values shalls be numpy arrays (ndarray) of type float32 or int32!")
    except:
        raise SystemExit("Can not read parameter_dict: {}".format(parameter_dict))

    if block_id_and_param_type is not None:
        blkIdParamTypeOk = nnc_core.nnr_model.sanity_check_block_id_and_param_type( block_id_and_param_type, parameter_dict )
        if blkIdParamTypeOk:
            nnc_core.nnr_model.set_block_id_and_param_type( nnc_mdl.model_info , block_id_and_param_type )
        else:
            print("INFO: Sanity check for block_id_and_param_type failed! block_id_and_param_type has been set to 'None', and the flags 'lsa' and 'bnf' have been set to 'False'!")
            block_id_and_param_type = None
            lsa = False
            bnf = False
            
    
    if model_executer:
        if lsa and not model_executer.has_tune_lsa():
            print("INFO: Tuning (training) of LSA parameters (tune_model) not implemented by model_executer! 'lsa' has been set to 'False'!")
            lsa = False
        if fine_tune and not model_executer.has_tune_ft():
            print("INFO: Fine tuning (training) of parameters (tune_model) not implemented by model_executer! 'fine_tune' has been set to 'False'!")
            fine_tune = False
        if ioq and not model_executer.has_eval():
            print("INFO: Evaluation (inference on a reduced dataset) of parameters (eval_model) not implemented by model_executer! ioq' has been set to 'False'!")
            ioq = False
                    
    ##INITIALIZATION
    approx_data =  nnc_core.approximator.init_approx_data(  model_parameters,
                                                            nnc_mdl.model_info, 
                                                            qp_density=qp_density, 
                                                            scan_order=scan_order
                                                         )

    ApproxInfoO = nnc_core.approximator.ApproxInfo( approx_data,
                                                    nnc_mdl.model_info,
                                                    "uniform" if codebook_mode==0 else "codebook",
                                                    codebook_mode,
                                                    qp,
                                                    opt_qp,
                                                    not use_dq,
                                                    cabac_unary_length_minus1,
                                                    lambda_scale,
                                                    nonweight_qp=nonweight_qp,
                                                    qp_per_tensor=qp_per_tensor
                                                )
    approx_info = ApproxInfoO.approx_info

    enc_info = {
            "cabac_unary_length_minus1" : cabac_unary_length_minus1,
            "param_opt_flag"     : param_opt,
        }
    end = timer()
    __print_output_line("DONE in {:.4f} s\n".format(end-start), verbose=verbose)

    ##PREPROCESSING
    if ioq:
        assert model_executer is not None, "model_executer must be available in order to run IOQ!"
        start = timer()
        __print_output_line("PREPROCESSING, IOQ...\n", verbose=verbose) 
        nnc_core.approximator.inference_based_qp_opt(
            approx_info,
            nnc_mdl.model_info,
            model_executer,
            approx_data,
            enc_info["param_opt_flag"],
            enc_info["cabac_unary_length_minus1"],
            verbose=verbose,
        )
        end = timer()
        __print_output_line("DONE in {:.4f} s\n".format( end-start ), verbose=verbose)   

    ##LSA and FT
    if lsa or fine_tune:
        assert model_executer is not None, "model_executer must be available in order to run LSA and/or FT!"
        start = timer()
        __print_output_line("PREPROCESSING, LSA/FT...\n", verbose=verbose) 
        nnc_core.approximator.run_ft_and_lsa(
            nnc_mdl.model_info,
            approx_data,
            ApproxInfoO,
            model_executer,
            block_id_and_param_type,
            lsa,
            fine_tune,
            use_dq,
            verbose
        )
        end = timer()
        __print_output_line("DONE in {:.4f} s\n".format( end-start ), verbose=verbose)  
    ##BNF
    if bnf:
        start = timer()
        __print_output_line("PREPROCESSING, BNF...", verbose=verbose)    
        nnc_core.approximator.fold_bn(nnc_mdl.model_info, approx_data, ApproxInfoO)
        end = timer()
        __print_output_line("DONE in {:.4f} s\n".format(end-start), verbose=verbose)

    #####QUANTIZATION AND ENCODING
    start = timer() 
    __print_output_line("APPROXIMATING WITH METHOD {}...".format(approx_info["approx_method"]), verbose=verbose)
    approx_data_enc = nnc_core.approximator.approx( approx_info,
                                                nnc_mdl.model_info,
                                                approx_data,
                                                enc_info["param_opt_flag"]
                                               )
    end = timer()
    __print_output_line("DONE in {:.4f} s\n".format( end-start ), verbose=verbose)

    start = timer()
    __print_output_line("ENCODING...", verbose=verbose)
    bitstream = nnc_core.coder.encode(  enc_info, 
                                    nnc_mdl.model_info, 
                                    approx_data_enc
                                 )
    end = timer()
    __print_output_line("DONE in {:.4f} s\n".format( end-start ), verbose=verbose)

    original_size = nnc_mdl.model_info["original_size"]

    __print_output_line("COMPRESSED FROM {} BYTES TO {} BYTES ({} KB, {} MB, COMPRESSION RATIO: {:.2f} %) in {:.4f} s\n".format(original_size, len(bitstream), len(bitstream)/1000.0, len(bitstream)/1000000.0, len(bitstream)/original_size*100, end-start_overall), verbose=verbose)
    
    if bitstream_path is not None:
        with open( bitstream_path, "wb" ) as br_file:
            br_file.write( bitstream )

    if return_bitstream:
        return bitstream


def decompress( bitstream_or_path, 
                block_id_and_param_type=None, 
                return_model_information=False, 
                verbose=True, 
                reconstruct_lsa=True, 
                reconstruct_bnf=True
                ):

    dec_model_info  = {'parameter_type': {},
                      'parameter_dimensions': {},
                      'parameter_index': {},
                      'block_identifier': {},
                      'topology_storage_format' : None,
                      'topology_compression_format' : None,
                      'performance_maps' : { "mps" : {}, "lps" : {}},
                      'performance_map_flags' : { "mps_sparsification_flag" : {}, "lps_sparsification_flag" : {},
                                                  "mps_pruning_flag" : {}, "lps_pruning_flag" : {},
                                                  "mps_unification_flag" : {}, "lps_unification_flag" : {},
                                                  "mps_decomposition_performance_map_flag" : {}, "lps_decomposition_performance_map_flag" : {},
                                                } 
                      }

    model_information = { 'topology_storage_format' : None,
                          'performance_maps' : {},
                          'performance_map_flags' : {}
                        }

    if block_id_and_param_type is not None:
        blkIdParamTypeOk = nnc_core.nnr_model.sanity_check_block_id_and_param_type( block_id_and_param_type )
        if blkIdParamTypeOk == False:
            print("INFO: Sanity check for block_id_and_param_type failed! block_id_and_param_type has been set to 'None'!")
            block_id_and_param_type = None
        else:
            nnc_core.nnr_model.set_block_id_and_param_type( dec_model_info, block_id_and_param_type )

    hls_bytes = {}
    start = timer()
    __print_output_line("DECODING...", verbose=verbose)
    if isinstance(bitstream_or_path, bytearray):
        bitstream = bitstream_or_path
    elif os.path.exists(os.path.expanduser(bitstream_or_path)):
        with open( os.path.expanduser(bitstream_or_path), "rb" ) as br_file:
            bitstream = br_file.read()
    else:
        raise SystemExit( "Could not read bitstream or bitstream_path: {}".format(bitstream_or_path) )

    dec_approx_data = nnc_core.coder.decode(bitstream, dec_model_info, hls_bytes)
    end = timer()
    __print_output_line("DONE in {:.4f} s\n".format( end-start ), verbose=verbose)

    start = timer()
    rec_approx_data = dec_approx_data
    __print_output_line("RECONSTRUCTING...", verbose=verbose)
    nnc_core.approximator.rec(rec_approx_data )
    if reconstruct_bnf:
        nnc_core.approximator.unfold_bn(dec_model_info, rec_approx_data)
    if reconstruct_lsa:
        nnc_core.approximator.apply_lsa(dec_model_info, rec_approx_data)
    rec_approx_data = nnc_core.approximator.recompose_params( dec_model_info, rec_approx_data)
    end = timer()
    __print_output_line("DONE in {:.4f} s\n".format( end-start ), verbose=verbose)
    
    if return_model_information:
        model_information["topology_storage_format"] = dec_model_info["topology_storage_format"]
        model_information["performance_maps"]        = dec_model_info["performance_maps"]
        model_information["performance_map_flags"]   = dec_model_info["performance_map_flags"]

        return rec_approx_data["parameters"], model_information
    else:
        return rec_approx_data["parameters"]


def decompress_model( bitstream_or_path,
                      model_path=None,#"./rec.mdl",
                      block_id_and_param_type=None,
                      model_struct=None,
                      model_executer=None,
                      model_name=None, 
                      dataset_path=None, 
                      batch_size=64, 
                      num_workers=8,
                      reconstruct_bnf=True,
                      reconstruct_lsa=True,
                      test_model=False,
                      return_model_information=False,
                      return_decompressed_model=False,
                      verbose=True,
                    ):
    
    if block_id_and_param_type is not None:
        blkIdParamTypeOk = nnc_core.nnr_model.sanity_check_block_id_and_param_type( block_id_and_param_type )
        if blkIdParamTypeOk == False:
            print("INFO: Sanity check for block_id_and_param_type failed! block_id_and_param_type has been set to 'None'!")
            block_id_and_param_type = None
    
    model_dict, model_information = decompress(bitstream_or_path, 
                                        block_id_and_param_type=block_id_and_param_type, 
                                        return_model_information=True,
                                        reconstruct_lsa=reconstruct_lsa,
                                        reconstruct_bnf=reconstruct_bnf
                                       )

    model_with_decoded_parameters = None

    if model_information["topology_storage_format"] == nnc_core.nnr_model.TopologyStorageFormat.NNR_TPL_PYT:
        if model_path == None:
            model_path="./rec.pt"

        pytorch_model.save_to_pytorch_file( model_dict, model_path )

        if ( (model_struct and dataset_path) or model_executer ) and test_model:
            
            if model_executer:
                nnc_mdl_executer = model_executer
            else:    
                _, nnc_mdl_executer, _ = pytorch_model.create_NNC_model_instance_from_file(
                    model_path,
                    dataset_path=dataset_path,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    model_struct=model_struct,
                    )
            
            acc = nnc_mdl_executer.test_model(
                    model_dict,
                    verbose=verbose
                    )
            print(acc)
        
        if model_struct and return_decompressed_model:
            model_with_decoded_parameters = pytorch_model.get_model_file_with_parameters(model_struct=model_struct, parameters=model_dict)
                
    elif model_information["topology_storage_format"] == nnc_core.nnr_model.TopologyStorageFormat.NNR_TPL_TEF:
        if model_path == None:
            model_path="./rec.h5"

        tensorflow_model.save_to_tensorflow_file( model_dict, model_path )
        
        if ( (model_struct and dataset_path) or model_executer ) and test_model:
            
            if model_executer:
                nnc_mdl_executer = model_executer
            else:
                _, nnc_mdl_executer, _ = tensorflow_model.create_NNC_model_instance_from_file(
                    model_path,
                    dataset_path=dataset_path,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    model_struct=model_struct,
                    model_name=model_name 
                    )

            acc = nnc_mdl_executer.test_model(
                    model_dict,
                    verbose=verbose
                    )
            print(acc)

        if model_struct and return_decompressed_model:
           model_with_decoded_parameters = tensorflow_model.get_model_file_with_parameters(model_struct=model_struct, parameters=model_dict)

    elif model_information["topology_storage_format"] == nnc_core.nnr_model.TopologyStorageFormat.NNR_TPL_UNREC or model_information["topology_storage_format"] == None:
        if model_path == None:
            model_path="./rec.mdl"

        nnr_model.save_to_pickled_file( model_dict, model_path )

        if model_executer and test_model:
            nnc_mdl_executer = model_executer
            acc = nnc_mdl_executer.test_model(
                model_dict,
                verbose=verbose
            )
            print(acc)

    else:
        raise SystemExit( "Topology Storage Format not yet supported: {}".format( model_information["topology_storage_format"] ) )
    
    if return_decompressed_model and return_model_information:
        return model_with_decoded_parameters, model_information
    elif return_decompressed_model:
        return model_with_decoded_parameters
    elif return_model_information:
        return model_information
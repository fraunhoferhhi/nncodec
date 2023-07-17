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
import sys
import numpy as np
from . import baseline
from . import codebook
from . import integer
import nnc_core
from nnc_core.nnr_model import NNRModelAccess, NNRBlockAccess, W_TYPES
from nnc_core import hls
from timeit import default_timer as timer

def __print_output_line( outputString, verbose=True ):
    if verbose:
        sys.stdout.write(outputString)
        sys.stdout.flush()

def del_param(approx_data, approx_info, param):
    del approx_data["parameters"][param]
    approx_data["scan_order"].pop(param, None)
    approx_info.get("qp", {}).pop(param, None)
    approx_info.get("dq_flag", {}).pop(param, None)
    
def init_approx_data(
    parameters,
    model_info,
    qp_density,
    scan_order,
): 
    approx_data = {
        "approx_method": {},
        "qp_density": np.int32(qp_density),
        "qp": {},
        "dq_flag": {},
        "decomposition_rank": {},
        "g_number_of_rows": {},
        "scan_order": {},
        "parameters": copy.copy(parameters),
        "compressed_parameter_types": {},
        "codebooks": {},
        "codebooks_egk": {},
        "codebook_zero_offsets": {},
    }
    for x in parameters:
        assert (x.endswith("_G") or x.endswith("_H")) == (("_G" in x) or ("_H" in x))
        if x.endswith("_G") or x.endswith("_H"):
            if len(model_info["parameter_dimensions"][x[:-2]]) > 1:
                approx_data["scan_order"][x] = np.int32(scan_order) 
        elif len(model_info["parameter_dimensions"][x]) > 1:
            approx_data["scan_order"][x] = np.int32(scan_order) 
        else:
            continue

    for block_id in model_info["block_identifier"].values():
        if block_id != None:
            block_access = NNRBlockAccess(model_info, block_id)
            cpt = 0
            if block_access.bn_gamma:
                cpt += hls.BlockParameterTypes.NNR_CPT_BN
            if block_access.bi in approx_data["parameters"]: 
                cpt += hls.BlockParameterTypes.NNR_CPT_BI
            if block_access.dc_g in approx_data["parameters"]: 
                cpt += hls.BlockParameterTypes.NNR_CPT_DC
                par_dc_g = block_access.dc_g
                #par_dc_h = block_access.dc_h
                approx_data["decomposition_rank"][block_id] = approx_data["parameters"][par_dc_g].shape[1] 
                approx_data["g_number_of_rows"][block_id] = approx_data["parameters"][par_dc_g].shape[0] 
            if block_access.ls in approx_data["parameters"]: 
                cpt += hls.BlockParameterTypes.NNR_CPT_LS
            approx_data["compressed_parameter_types"][block_id] = cpt

    return approx_data

        
def fold_bn(model_info, approx_data, ap_info):
    model_access = NNRModelAccess(model_info)
    for block_access in model_access.blocks_and_params():
        block_id = block_access.block_id
        if block_id is None:
            continue
        cpt = approx_data["compressed_parameter_types"][block_id]
        ad = approx_data["parameters"]
        assert not approx_data["approx_method"]
        eps = 1e-3 if model_info['topology_storage_format'] == nnc_core.nnr_model.TopologyStorageFormat.NNR_TPL_TEF else 1e-5
        if cpt & hls.BlockParameterTypes.NNR_CPT_BN != 0:
            delta = block_access.bi
            bn_shape = ad[block_access.bn_mean].shape
            dq_flag = ap_info.approx_info["dq_flag"][block_access.bn_mean]
            assert (cpt & hls.BlockParameterTypes.NNR_CPT_BI == 0) == (delta not in ad)
            if cpt & hls.BlockParameterTypes.NNR_CPT_BI == 0:
                ad[delta] = np.zeros(bn_shape, dtype=np.float32)
                approx_data["compressed_parameter_types"][block_id] += hls.BlockParameterTypes.NNR_CPT_BI
                if ap_info.approx_info["approx_method"] == "uniform":
                    ap_info.approx_info["qp"][delta] = ap_info.qp_other
                    ap_info.approx_info["dq_flag"][delta] = dq_flag
            alpha = block_access.ls
            assert (cpt & hls.BlockParameterTypes.NNR_CPT_LS == 0) == (alpha not in ad)
            if cpt & hls.BlockParameterTypes.NNR_CPT_LS == 0:
                assert bn_shape == ad[block_access.bn_mean].shape
                ad[alpha] = np.ones(bn_shape, dtype=np.float32)
                approx_data["compressed_parameter_types"][block_id] += hls.BlockParameterTypes.NNR_CPT_LS
                if ap_info.approx_info["approx_method"] == "uniform":
                    ap_info.approx_info["qp"][alpha] = ap_info.qp_lsa
                    ap_info.approx_info["dq_flag"][alpha] = dq_flag

            g = ad[block_access.bn_gamma] / np.sqrt( ad[block_access.bn_var] + eps )
            del_param(approx_data, ap_info.approx_info, block_access.bn_gamma)
            del_param(approx_data, ap_info.approx_info, block_access.bn_var)
            ad[alpha] *= g
            ad[delta] = (ad[delta] - ad[block_access.bn_mean]) * g + ad[block_access.bn_beta]
            del_param(approx_data, ap_info.approx_info, block_access.bn_mean)
            del_param(approx_data, ap_info.approx_info, block_access.bn_beta)
            approx_data["compressed_parameter_types"][block_id] -= hls.BlockParameterTypes.NNR_CPT_BN

            
def unfold_bn(model_info, approx_data):
    model_access = NNRModelAccess(model_info)
    ad = approx_data["parameters"]
    for block_access in model_access.blocks_and_params():
        block_id = block_access.block_id
        if block_id is None:
            continue

        bn_absent = approx_data["compressed_parameter_types"][block_id] & hls.BlockParameterTypes.NNR_CPT_BN == 0
        bn_folded = bn_absent and (block_access.bn_gamma in model_info["parameter_type"])
        if bn_folded:
            approx_data["compressed_parameter_types"][block_id] += hls.BlockParameterTypes.NNR_CPT_BN
            #ad = approx_data["parameters"]

            delta = block_access.bi
            dims = approx_data["parameters"][delta].shape
            if delta not in model_info["parameter_type"]:
                assert approx_data["compressed_parameter_types"][block_id] & hls.BlockParameterTypes.NNR_CPT_BI != 0
                approx_data["parameters"][block_access.bn_beta] = approx_data["parameters"][delta]
                del approx_data["parameters"][delta]
                approx_data["compressed_parameter_types"][block_id] -= hls.BlockParameterTypes.NNR_CPT_BI
            else:
                approx_data["parameters"][block_access.bn_beta] = np.zeros(dims, dtype=np.float32)

            approx_data["parameters"][block_access.bn_mean]  = np.zeros(dims, dtype=np.float32)
            approx_data["parameters"][block_access.bn_gamma] = np.ones(dims, dtype=np.float32)
            approx_data["parameters"][block_access.bn_var]   = np.ones(dims, dtype=np.float32)


def set_lsa(model_info, approx_data, lsa_params):
    for k, v in lsa_params.items():
        approx_data["parameters"][k] = v.reshape([v.shape[0]])
        bi = model_info["block_identifier"].get(k, None)
        if bi is not None:
            approx_data["compressed_parameter_types"][bi] |= hls.BlockParameterTypes.NNR_CPT_LS

def apply_lsa(model_info, approx_data):
    assert not approx_data["approx_method"]
    model_access = NNRModelAccess(model_info)
    for block_access in model_access.blocks_and_params():
        block_id = block_access.block_id
        if block_id is None:
            continue
        cpt = approx_data["compressed_parameter_types"][block_id]
        if cpt & hls.BlockParameterTypes.NNR_CPT_LS != 0:
            ls = approx_data['parameters'].pop(block_access.ls)
            _ = model_info["parameter_index"].pop(block_access.ls, None)
            _ = model_info["block_identifier"].pop(block_access.ls, None)

            if cpt & hls.BlockParameterTypes.NNR_CPT_DC != 0:
                w = approx_data['parameters'][block_access.dc_g]
            else:
                w = approx_data['parameters'][block_access.w]
            dims_ls = [-1] + [1] * (w.ndim - 1)
            w *= ls.reshape(dims_ls)
            approx_data['compressed_parameter_types'][block_id] -= hls.BlockParameterTypes.NNR_CPT_LS

def recompose_params(model_info, approx_data_in):
    assert not approx_data_in["approx_method"]
    approx_data_out = {k: copy.copy(v) for k, v in approx_data_in.items()} # create copies of dicts in approx_data

    model_access = NNRModelAccess(model_info)
    for block_access in model_access.blocks_and_params():
        block_id = block_access.block_id
        if block_id is None:
            continue
        cpt = approx_data_out["compressed_parameter_types"][block_id]
        if cpt & hls.BlockParameterTypes.NNR_CPT_DC != 0:
            g = approx_data_out["parameters"].pop(block_access.dc_g)
            h = approx_data_out["parameters"].pop(block_access.dc_h)

            recomposed_w = g.dot(h)
            recomposed_w = recomposed_w.reshape(model_info["parameter_dimensions"][block_access.w])

            approx_data_out["parameters"][block_access.w] = recomposed_w
            approx_data_out['compressed_parameter_types'][block_id] -= hls.BlockParameterTypes.NNR_CPT_DC

            param_id_g = model_info["parameter_index"][block_access.dc_g].pop()
            model_info["parameter_index"][block_access.w] = param_id_g
            del(model_info["block_identifier"][block_access.dc_g])
            del(model_info["parameter_index"][block_access.dc_h])
            del(model_info["block_identifier"][block_access.dc_h])

    resorted_param_dict = dict()
    resorted_param_id_dict = {k: v for k, v in sorted(model_info["parameter_index"].items(), key=lambda item: item[1])}
    for param in resorted_param_id_dict.keys():
        resorted_param_dict[param] = copy.deepcopy(approx_data_out["parameters"][param])

    approx_data_out["parameters"] = resorted_param_dict

    return approx_data_out


def inference_based_qp_opt( 
        approx_info,
        model_info,
        model_executer,
        approx_data,
        param_opt,
        cabac_unary_length_minus1,
        verbose,
    ):
    approx_data_qp = approx(
        approx_info,
        model_info,
        approx_data,
        param_opt,
    )
    rec_approx_data_qp = copy.deepcopy(approx_data_qp)
    rec(
        rec_approx_data_qp,
    )
    
    ##encode
    start = timer()
    __print_output_line("\tIOQ: PROCESSING QP FOR ALL TENSORS...", verbose=verbose) 
    enc_info_qp = {
        "cabac_unary_length_minus1" : cabac_unary_length_minus1,
        "param_opt_flag" : param_opt,
    }
    bitstream_qp = nnc_core.coder.encode(enc_info_qp, model_info, approx_data_qp)

    acc_qp = model_executer.eval_model(
        rec_approx_data_qp["parameters"],
        False,
    )
    
    refBSSize = len(bitstream_qp)
    refAcc = acc_qp[0]

    bestCost = 0.0
    end = timer()
    __print_output_line("DONE in {:.4f} s\n".format( end-start ), verbose=verbose) 


    ############################ eval with QP-1
    start = timer()
    __print_output_line("\tIOQ: PROCESSING QP-1 FOR ALL TENSORS...", verbose=verbose) 
 
    approx_info_qp = copy.deepcopy(approx_info)

    for p in approx_info_qp["qp"].keys():
        if model_info["parameter_type"][p] in nnc_core.nnr_model.W_TYPES:
            approx_info_qp["qp"][p] -= 1

    approx_data_qp = approx(
        approx_info_qp,
        model_info,
        approx_data,
        param_opt,
    )
    rec_approx_data_qp = copy.deepcopy(approx_data_qp)
    rec(
        rec_approx_data_qp,
    )
    
    ##encode
    enc_info_qp = {
        "cabac_unary_length_minus1" : cabac_unary_length_minus1,
        "param_opt_flag" : param_opt,
    }
    bitstream_qp = nnc_core.coder.encode(enc_info_qp, model_info, approx_data_qp)
    ##eval
    acc_qp = model_executer.eval_model(
        rec_approx_data_qp["parameters"],
        False,
    )

    currBSSize = len(bitstream_qp)
    currAcc = acc_qp[0]

    diffBR = currBSSize - refBSSize
    diffAcc = refAcc - currAcc

    lambdaM1 = -diffAcc/diffBR 

    end = timer()
    __print_output_line("DONE in {:.4f} s\n".format( end-start ), verbose=verbose)  


    ############################### eval with QP+1
    start = timer()
    __print_output_line("\tIOQ: PROCESSING QP+1 FOR ALL TENSORS...", verbose=verbose) 

    approx_info_qp = copy.deepcopy(approx_info)

    for p in approx_info_qp["qp"].keys():
        if model_info["parameter_type"][p] in nnc_core.nnr_model.W_TYPES:
            approx_info_qp["qp"][p] += 1

    approx_data_qp = approx(
        approx_info_qp,
        model_info,
        approx_data,
        param_opt,
    )
    rec_approx_data_qp = copy.deepcopy(approx_data_qp)
    rec(
        rec_approx_data_qp,
    )
    
    ##encode
    enc_info_qp = {
        "cabac_unary_length_minus1" : cabac_unary_length_minus1,
        "param_opt_flag" : param_opt,
    }
    bitstream_qp = nnc_core.coder.encode(enc_info_qp, model_info, approx_data_qp)
    ##eval
    acc_qp = model_executer.eval_model(
        rec_approx_data_qp["parameters"],
        False,
    )

    currBSSize = len(bitstream_qp)
    currAcc = acc_qp[0]

    diffBR = currBSSize - refBSSize
    diffAcc = refAcc - currAcc

    lambdaP1 = -diffAcc/diffBR
    
    end = timer()
    __print_output_line("DONE in {:.4f} s\n".format( end-start ), verbose=verbose) 

    ################################

    ##sort parameters by size
    mapParamToSize = []
    approx_info_qp = copy.deepcopy(approx_info)
    for p in rec_approx_data_qp["parameters"]:
        if model_info["parameter_type"][p] in nnc_core.nnr_model.W_TYPES:
            mapParamToSize.append([p , np.size(approx_data_qp["parameters"][p])])
    
    mapParamToSize.sort(key = lambda x: x[1],reverse=True) 
    
    setNeg = [-4, -3, -2, -1 ]
    setPos = [1, 2, 3, 4 ]

    qpOffsetSet = [setNeg, setPos]

    timeLastQp = "n/a"

    for iParam, item in enumerate(mapParamToSize[1::]):
        for iQpSet, qpSet in enumerate(qpOffsetSet):
            for iQpOff, qp_off in enumerate(qpSet):
                __print_output_line("\r\tIOQ: PROCESSING TENSOR {}/{} AND QP {}/{} (LAST QP-ITERATION TOOK: {})".format( iParam+1, len(mapParamToSize)-1, iQpSet*len(qpOffsetSet[0])+ iQpOff + 1, len(qpOffsetSet[0])+len(qpOffsetSet[1]), timeLastQp), verbose=verbose)
                start = timer()
                approx_info_qp_curr = copy.deepcopy(approx_info_qp)
                approx_info_qp_curr["qp"][item[0]] = approx_info["qp"][item[0]] + qp_off

                approx_data_qp = approx(
                    approx_info_qp_curr,
                    model_info,
                    approx_data,
                    param_opt,
                )
                    
                rec_approx_data_qp = copy.deepcopy(approx_data_qp)
                rec(
                    rec_approx_data_qp,
                )
                
                ##encode
                enc_info_qp = {
                    "cabac_unary_length_minus1" : cabac_unary_length_minus1,
                    "param_opt_flag" : param_opt,
                }
                
                bitstream_qp = nnc_core.coder.encode(enc_info_qp, model_info, approx_data_qp)
            
                ##eval
                acc_qp = model_executer.eval_model(
                    rec_approx_data_qp["parameters"],
                    False,
                )
                
                currBSSize = len(bitstream_qp)
                currAcc = acc_qp[0]

                diffBR = currBSSize - refBSSize
                diffAcc = refAcc - currAcc

                lamb = max( (lambdaP1 + lambdaM1) / 2, 0.0 )

                currCost = diffAcc + lamb * diffBR

                if currCost < bestCost:
                    approx_info_qp = copy.deepcopy(approx_info_qp_curr)
                    bestCost = currCost
                
                end = timer()
                timeLastQp = "{:.4f} s".format( end-start )

    __print_output_line("\n")
    approx_info.clear()
    approx_info.update(approx_info_qp)


def run_ft_and_lsa(model_info, approx_data, ap_info, model_executer, block_id_and_param_type, lsa_flag, ft_flag, use_dq, verbose):
    approx_info_ft = copy.deepcopy(ap_info.approx_info)
    if not lsa_flag:
        approx_info_ft["to_approximate"] = W_TYPES
    else:
        approx_info_ft["to_approximate"].remove('weight.ls')
    approx_data_ft = nnc_core.approximator.approx(approx_info_ft, model_info, approx_data)
    nnc_core.approximator.rec(approx_data_ft)

    tuned_params = model_executer.tune_model(
        parameters=approx_data_ft['parameters'],
        param_types=model_info['parameter_type'],
        lsa_flag=lsa_flag,
        ft_flag=ft_flag,
        verbose=verbose,
    )
    
    lsa_params = tuned_params[0]
    ft_params  = tuned_params[1]

    if ft_flag:
        approx_data["parameters"].update(ft_params)
    if lsa_flag:
        if block_id_and_param_type:
            nnc_core.approximator.set_lsa(model_info, approx_data, lsa_params)
            nnc_core.nnr_model.add_lsa_to_block_id_and_param_type( block_id_and_param_type, lsa_params )
        else:
            approx_data["parameters"].update(lsa_params)
        ap_info.set_ls_qps(model_info, approx_data, 1 if use_dq else 0)


def approx(approx_info, model_info, approx_data, param_opt=0):
    approx_method = approx_info['approx_method']
    
    approx_data = integer.skip_approx( approx_info, model_info, approx_data )

    if approx_method == 'codebook':
        approx_data, approx_info = codebook.approx(approx_info, model_info, approx_data, param_opt)

    return baseline.approx(approx_info, model_info, approx_data)
    

def rec(approx_data):
    for param in approx_data['parameters']:
        if param in approx_data["approx_method"]:
            if approx_data["approx_method"][param] == 'uniform':
                baseline.rec(param, approx_data)
            elif approx_data["approx_method"][param] == 'codebook':
                codebook.rec(param, approx_data)
            elif approx_data["approx_method"][param] == 'skip':
                integer.skip_rec(param, approx_data)
            else:
                assert param not in approx_data["approx_method"], "unknown approx_method"


class ApproxInfo():
    def __init__(
        self,
        approx_data,
        model_info,
        approx_method,
        codebook_mode,
        qp,
        opt_qp,
        disable_dq,
        cabac_unary_length_minus1,
        lambda_scale,
        nonweight_qp=None,
        qp_per_tensor=None,
    ):
        self.__approx_info = {
            "approx_method": "codebook" if codebook_mode > 0 else approx_method,
            "codebook_mode": codebook_mode,
            "dq_flag": {x: 0 if disable_dq else 1 for x in approx_data["parameters"]},
            "lambda_scale": lambda_scale,
            "cabac_unary_length_minus1": cabac_unary_length_minus1,
            "to_approximate": nnc_core.nnr_model.W_TYPES + nnc_core.nnr_model.O_TYPES,
        }

        if approx_method == "uniform" or approx_method == "codebook":
            qp = np.int32(qp)
            qp_density = approx_data["qp_density"]
            self.__qp_other = nonweight_qp if nonweight_qp else qp - (2 << qp_density)  # same as dividing the stepsize by 4
            self.__qp_lsa = nonweight_qp if nonweight_qp else qp - (2 << qp_density)#qp - (8 << qp_density)
            self.approx_info["qp"] = {}
            for x in approx_data["parameters"]:
                if x not in model_info["parameter_index"] and (x.endswith("_G") or x.endswith("_H")):
                    assert model_info["parameter_type"][x[:-2]] in nnc_core.nnr_model.W_TYPES, "Unexpected."
                    self.approx_info["qp"][x] = qp
                else:
                    self.approx_info["qp"][x] = qp if model_info["parameter_type"][x] in nnc_core.nnr_model.W_TYPES else self.qp_other
            if qp_per_tensor is not None:
                assert type(qp_per_tensor) is dict, "qp_per_tensor must be a dict!"  
                for x in approx_data["parameters"]:
                    self.approx_info["qp"][x] = qp_per_tensor.get(x, self.approx_info["qp"][x])
            if opt_qp:
                self._modify_qp(approx_data, model_info)

    @property
    def qp_lsa(self):
        return self.__qp_lsa

    @property
    def qp_other(self):
        return self.__qp_other

    @property
    def approx_info(self):
        return self.__approx_info

    def apply_qp(self, approx_data, model_info, qp, nonweight_qp=None):
        qp = np.int32(qp)
        qp_density = approx_data["qp_density"]
        self.__qp_other = nonweight_qp if nonweight_qp else qp - (2 << qp_density)
        self.__qp_lsa = nonweight_qp if nonweight_qp else qp - (2 << qp_density)#qp - (8 << qp_density)
        self.approx_info["qp"] = {}
        for x in approx_data["parameters"]:
            if x not in model_info["parameter_index"] and (x.endswith("_G") or x.endswith("_H")):
                assert model_info["parameter_type"][x[:-2]] in nnc_core.nnr_model.W_TYPES, "Unexpected."
                self.approx_info["qp"][x] = qp
            else:
                if model_info["parameter_type"][x] in nnc_core.nnr_model.W_TYPES:
                    self.approx_info["qp"][x] = qp
                else:
                    self.approx_info["qp"][x] = self.qp_other
    
    def _modify_qp(self, approx_data, model_info):
        param_types = ["weight"]#["fc.weight", "conv.weight"]
        param_names = []
        param_sizes = []
        param_std = []
        for k, v in approx_data["parameters"].items():
            param_w = k[:-2] if ( k.endswith("_G") or k.endswith("_H") ) else k 
            if model_info["parameter_type"][param_w] in param_types:
                if ( k.endswith("_G") or k.endswith("_H") ):
                    if k.endswith("_G") : continue
                    assert k.endswith("_H")
                    g = approx_data["parameters"][param_w + "_G"].shape
                    h = approx_data["parameters"][param_w + "_H"].shape
                    if len( h ) == 4:
                        assert h[0] == 1
                        assert h[1] == 1
                    assert h[-2] == g[-1]
                    s = np.prod(g[:-1]) * h[-1]
                    param_names.append(param_w + "_G")
                    param_sizes.append(0)
                    param_std.append(0)
                    param_names.append(param_w + "_H")
                    param_sizes.append(s)
                    param_std.append(np.std(np.concatenate(
                        (approx_data["parameters"][param_w + "_G"].flatten(),
                         approx_data["parameters"][param_w + "_H"].flatten()), axis=0)))
                else:
                    param_names.append(k)
                    param_sizes.append(v.size)
                    param_std.append(np.std(v))

        rel_layer_sizes = np.array(param_sizes) / sum(param_sizes)
        rel_layer_std = np.array(param_std) / max(param_std)

        shares = rel_layer_sizes + (.1 * (1 - rel_layer_std))

        w = dict(zip(param_names, shares))
        for name in param_names:
            qp = self.__approx_info['qp'][name]
            if w[name] > .5: w[name] = .15
            self.__approx_info['qp'][name] = np.int32(round(qp * (1 - w[name])))
            if name.endswith( "_H" ):
                self.__approx_info['qp'][name[:-2]+"_G"] = self.__approx_info['qp'][name]

    def set_ls_qps(self, model_info, approx_data, dq_flag):
        for block_access in NNRModelAccess(model_info).blocks_and_params():
            if block_access.block_id is not None:
                cpt = approx_data["compressed_parameter_types"][block_access.block_id]
                if cpt & hls.BlockParameterTypes.NNR_CPT_LS != 0:
                    self.approx_info["qp"][block_access.ls] = self.qp_lsa
                    self.approx_info["dq_flag"][block_access.ls] = dq_flag
            


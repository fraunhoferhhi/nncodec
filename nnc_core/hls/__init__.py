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

import enum
import numpy as np
import sys
from nnc_core import nnr_model

assert sys.byteorder == "little"


class NnrUnitType(enum.IntEnum):
    NNR_STR = 0
    NNR_MPS = 1
    NNR_LPS = 2
    NNR_TPL = 3
    NNR_QNT = 4
    NNR_NDU = 5
    NNR_AGG = 6

    
class DecompressedDataFormat(enum.IntEnum):
    TENSOR_INT32   = 0
    TENSOR_FLOAT32 = 1

    
class CompressedDataUnitPayloadType(enum.IntEnum):
    NNR_PT_INT       = 0 ##tdb not yet implemented
    NNR_PT_FLOAT     = 1
    NNR_PT_RAW_FLOAT = 2
    NNR_PT_BLOCK     = 3

    
class BlockParameterTypes(enum.IntEnum):
    NNR_CPT_DC = 0x01
    NNR_CPT_LS = 0x02
    NNR_CPT_BN = 0x04
    NNR_CPT_BI = 0x08

    
class QuantizationMethodFlags(enum.IntEnum):
    NNR_QSU = 1
    NNR_QCB = 2

class PruningInformationRepresentationTypes(enum.IntEnum):
    NNR_TPL_BMSK = 0
    NNR_TPL_DICT = 1
    
class BitWriter():
    def __init__( self, bitstream ):
        self.__byteList = bitstream
        self.__bitPos = -1
        self.__bytePos = -1
    
    def getNumBitsTouched(self):
        return (self.__bytePos + 1) * 8 - self.__bitPos - 1

    def writeBit(self, bit ):
        assert bit == 0 or bit == 1
        if self.__bitPos < 0:
            self.__bytePos += 1
            if len(self.__byteList) <= self.__bytePos:
                self.__byteList.append( 0 )
            self.__bitPos = 7
        if bit:
            self.__byteList[self.__bytePos] |= 1 << self.__bitPos
        else:
            self.__byteList[self.__bytePos] &= 255 - (1 << self.__bitPos)
        self.__bitPos -= 1

    def u(self, n, x):
        assert n > 0
        assert x >= 0 and x < (1<<n)
        for i in range( n-1, -1, -1):
            self.writeBit( (np.uint64(x) >> np.uint64(i)) & np.uint64(1) )

    def ue(self, k, x):
        while x >= (1<<k):
            self.u(1, 0)
            x -= 1<<k
            k +=1

        self.u(1, 1)
        if k > 0:
            self.u(k, x)

    def i(self, n, x):
        assert n > 0
        assert x >= -(1<<(n-1)) and x < (1<<(n-1))
        self.u( n, x if x >= 0 else x + (1<<n) )

    def ie(self, k, x):
        x = ((-x)<<1) if x <= 0 else ((x<<1)-1)
        self.ue(k,x)

    def byte_alignment(self):
        self.u(1,1)
        self.__bitPos = -1
        
    def flt(self, n, x):
        assert n == 32
        assert self.__bytePos + 1 == len(self.__byteList) # only supported at end of the bitstream
        assert isinstance( x, np.float32 )
        for byte in np.float32(x, dtype='<f4').tobytes():
            self.u(8, byte)

    def flt_tensor(self, n, dims, x):
        assert self.__bitPos == -1
        assert n == 32
        assert self.__bytePos + 1 == len(self.__byteList) # only supported at end of the bitstream
        assert isinstance( x, np.ndarray )
        assert x.dtype == np.float32
        self.__byteList.extend( x.tobytes() )
        self.__bytePos = len(self.__byteList) - 1

    def st(self, v):
        assert self.__bitPos == -1
        assert isinstance( v, str )
        assert self.__bytePos + 1 == len(self.__byteList) # only supported at end of the bitstream
        self.__byteList.extend( v.encode( "utf-8", "strict" ) )
        self.__byteList.append( 0 )
        self.__bytePos = len(self.__byteList) - 1

    def codebook(self, codebook_egk, codebook_size, CbZeroOffset, codebook):
        previousValue = codebook[CbZeroOffset]
        self.ie(7, previousValue) # codebook_zero_value
        for j in range(CbZeroOffset-1, -1, -1):
            self.ue(codebook_egk, previousValue - codebook[j] - 1 ) # codebook_delta_left
            previousValue = codebook[j]
        previousValue = codebook[CbZeroOffset]
        for j in range(CbZeroOffset+1, codebook_size):
            self.ue(codebook_egk, codebook[j] - previousValue - 1 ) # codebook_delta_right
            previousValue = codebook[j]

    def cbZeroOffset( self, codebook_size, CbZeroOffset ):
        codebook_center_offset = CbZeroOffset - (codebook_size >> 1)
        self.ie(2, codebook_center_offset)        

    def entry_point_list(self, block_rows_minus1, dq_flag, cabac_entry_point_list):
        for j in range(block_rows_minus1):
            ep = int(cabac_entry_point_list[j])
            bit_offset  =  ep >> 11
            value       = (ep >>  3) & 255
            dq_state    =  ep & 7
            self.u(8, value)
            if dq_flag:
                self.u(3, dq_state)
            if j == 0:
                self.ue(11, bit_offset)
            else:
                self.ie(7, bit_offset - (int(cabac_entry_point_list[j-1])>>11))

        
class BitReader():
    def __init__(self, byteList):
        assert isinstance( byteList, bytearray )
        self.__byteList = byteList
        self.__bitPos = -1
        self.__bytePos = -1
    
    def readBit(self):
        if self.__bitPos < 0:
            self.__bytePos += 1
            self.__bitPos = 7
        bit = (self.__byteList[self.__bytePos] >> self.__bitPos) & 1
        self.__bitPos -= 1
        return bit
    
    def getNumBytesTouched(self):
        return self.__bytePos + 1

    def getNumBitsTouched(self):
        return (self.__bytePos + 1) * 8 - self.__bitPos - 1

    def u(self, n):
        val = 0
        for i in range( n ):
            val += val + self.readBit()
        return val

    def ue(self, k):
        x = 0
        bit = 1
        while bit:
            bit = 1 - self.u(1)
            x += bit << k
            k += 1

        k -= 1
        if k > 0:
            x += self.u(k)
        return x

    def i(self, n):
        val = self.u( n )
        return val if val < (1<<(n-1)) else val - (1<<n)

    def ie(self,k):
        x = self.ue(k)
        return ((x+1)>>1) if (x&1) else -(x>>1)

    def byte_alignment(self):
        self.u(1)
        self.__bitPos = -1

    def flt(self, n ):
        assert n == 32
        dec_bytes = np.uint32(0)
        dec_bytes += self.u(8)
        dec_bytes += self.u(8) << 8
        dec_bytes += self.u(8) << 16
        dec_bytes += self.u(8) << 24
        z = np.frombuffer( dec_bytes.tobytes(), dtype='<f4', count=1 )
        return z

    def flt_tensor(self, n, dims ):
        assert self.__bitPos == -1
        assert n == 32
        count = np.prod(dims)
        z = np.frombuffer( self.__byteList[self.__bytePos+1:], dtype=np.float32, count=count ).reshape(dims)
        self.__bytePos += 4 * count 
        return z

    def st(self):
        assert self.__bitPos == -1
        remainingBytes = self.__byteList[self.__bytePos+1:]
        strLen = remainingBytes.find( 0 )
        self.__bytePos += strLen + 1
        return  remainingBytes[:strLen].decode( "utf-8", "strict" )

    def codebook(self, codebook_egk, codebook_size, CbZeroOffset):
        codebook = np.zeros(codebook_size, dtype=np.int32)
        previousValue = self.ie(7) # codebook_zero_value
        codebook[CbZeroOffset] = previousValue
        for j in range(CbZeroOffset-1, -1, -1):
            codebook[j] = previousValue - self.ue(codebook_egk) - 1 # codebook_delta_left
            previousValue = codebook[j]
        previousValue = codebook[CbZeroOffset]
        for j in range(CbZeroOffset+1, codebook_size):
            codebook[j] = self.ue(codebook_egk) + previousValue + 1 # codebook_delta_right
            previousValue = codebook[j]
        return codebook

    def cbZeroOffset( self, codebook_size ):
        codebook_center_offset = self.ie(2)
        CbZeroOffset = (codebook_size >> 1) + codebook_center_offset
        return CbZeroOffset
        
    def entry_point_list(self, block_rows_minus1, dq_flag):
        cabac_entry_point_list = np.zeros(block_rows_minus1, dtype=np.uint64)
        for j in range(block_rows_minus1):
            value       = self.u(8)
            dq_state = 0
            if dq_flag:
                dq_state    = self.u(3)
            if j == 0:
                bit_offset = self.ue(11)
            else:
                bit_offset = (int(cabac_entry_point_list[j-1])>>11) + self.ie(7)
            cabac_entry_point_list[j] = (bit_offset<<11) + (value<<3) + dq_state
        return cabac_entry_point_list

class Coder():
    def __init__(self, coder, seDict ):
        self.__seDict = seDict
        self.__coder = coder
        if isinstance( coder, BitReader ):
            self.__isReader = True
        elif isinstance( coder, BitWriter ):
            self.__isReader = False
        else:
            assert 0
    
    def get(self, se ):
        return self.__seDict[se]
    
    def define_array(self, se, dims, dtype ):
        if self.__isReader:
            assert se not in self.__seDict
            if dtype == str:
                assert len( dims ) == 1
                self.__seDict[se] = [ None for _ in range( dims[0] ) ]
            else:
                self.__seDict[se] = np.zeros( dims, dtype=dtype)

    def extend_second_2darray_dimension(self, se, ndl, dtype): ##extend second dimension of 2d array to new dimension ndl
        if self.__isReader:
            assert se in self.__seDict
            assert dtype != str
            assert len(self.__seDict[se].shape) == 2
            cs = self.__seDict[se].shape
            if ndl > cs[1]:
                self.__seDict[se] = np.append( self.__seDict[se], np.zeros( ( cs[0], ndl-cs[1] ), dtype=dtype ), axis=1 )

    def process(self, se, method, *args):
        se_split = se.split( "[" )
        array_idx = None
        array_idx2 = None
        if len( se_split ) > 1:
            se = se_split[0]
            array_idx = int( se_split[1].split("]")[0] )
        if len( se_split ) > 2:
            array_idx2 = int( se_split[2].split("]")[0] )
        m = getattr( self.__coder, method )
        if self.__isReader:
            if array_idx is not None and array_idx2 is not None:
                self.__seDict[se][array_idx][array_idx2] = m( *args )
            elif array_idx is not None:
                self.__seDict[se][array_idx] = m( *args )
            else:
                self.__seDict[se] = m( *args )
        else:
            if array_idx is not None and array_idx2 is not None:
                m( *args, self.__seDict[se][array_idx][array_idx2] )
            elif array_idx is not None:
                m( *args, self.__seDict[se][array_idx] )
            else:
                m( *args, self.__seDict[se] )
    
    def nnr_unit(self, numBytesInNNRUnit ):
        self.nnr_unit_size()
        yield from self.nnr_unit_header()
        self.nnr_unit_payload()

    def nnr_unit_size(self):
        self.process( "nnr_unit_size_flag", "u", 1 )
        self.process( "nnr_unit_size",      "u", 15 + 16 * self.get("nnr_unit_size_flag") )
        
    def nnr_unit_header(self):
        self.process( "nnr_unit_type",                "u", 6 )
        yield
        self.process( "independently_decodable_flag"     , "u", 1 )
        self.process( "partial_data_counter_present_flag", "u", 1)
        if self.get( "partial_data_counter_present_flag" ) == 1:
            self.process( "partial_data_counter", "u", 8 )
        if self.get( "nnr_unit_type" ) == NnrUnitType.NNR_MPS:
            self.nnr_model_parameter_set_unit_header()
        if self.get( "nnr_unit_type" ) == NnrUnitType.NNR_LPS:
            self.nnr_layer_parameter_set_unit_header()
        if self.get( "nnr_unit_type" ) == NnrUnitType.NNR_TPL:
            self.nnr_topology_unit_header()
        if self.get( "nnr_unit_type" ) == NnrUnitType.NNR_QNT:
            self.nnr_quanization_unit_header()
        if self.get( "nnr_unit_type" ) == NnrUnitType.NNR_NDU:
            yield from self.nnr_compressed_data_unit_header()
        if self.get( "nnr_unit_type" ) == NnrUnitType.NNR_STR:
            self.nnr_start_unit_header()
        if self.get( "nnr_unit_type" ) == NnrUnitType.NNR_AGG:
            self.nnr_aggregate_unit_header()

    def nnr_start_unit_header(self):
        self.process( "general_profile_idc", "u", 8 ) ##Not yet specified!

    def nnr_model_parameter_set_unit_header(self):
        ## nnr_reserved_zero_0bit, "u", 0
        pass
    
    def nnr_layer_parameter_set_unit_header(self): ##tbd: implement lps!
        self.process( "lps_self_contained_flag", "u", 1 )
        self.process( "nnr_reserved_zero_7bits", "u", 7 )

    def nnr_topology_unit_header(self):
        self.process( "topology_storage_format", "u", 8 )
        self.process( "topology_compression_format", "u", 8 )
    
    def nnr_quanization_unit_header(self):
        self.process( "quantization_storage_format", "u", 8 )
        self.process( "quantization_compression_format", "u", 8 )

    def codebook(self, cb_suffix):
        self.process( "codebook_egk__"+cb_suffix, "u", 4 )
        self.process( "codebook_size__"+cb_suffix, "ue", 2 )
        self.process( "CbZeroOffset__"+cb_suffix, "cbZeroOffset", self.get( "codebook_size__"+cb_suffix ) )
        self.process( "codebook__"+cb_suffix, "codebook", self.get( "codebook_egk__"+cb_suffix ), self.get( "codebook_size__"+cb_suffix ), self.get( "CbZeroOffset__"+cb_suffix ) )
        
    def nnr_compressed_data_unit_header(self):
        self.process( "nnr_compressed_data_unit_payload_type",       "u", 5 )
        self.process( "nnr_multiple_topology_elements_present_flag", "u", 1 )
        self.process( "nnr_decompressed_data_format_present_flag",   "u", 1 )
        self.process( "input_parameters_present_flag",               "u", 1 )
        
        if self.get("nnr_multiple_topology_elements_present_flag") ==  1:
            self.topology_elements_ids_list(self.get( "mps_topology_indexed_reference_flag" ))
        else:
            if self.get( "mps_topology_indexed_reference_flag" ) == 0: 
                self.process( "topology_elem_id", "st" )
            else:
                self.process( "topology_elem_id_index", "ue", 7 ) 

        if (
            (self.get( "nnr_compressed_data_unit_payload_type" ) == CompressedDataUnitPayloadType.NNR_PT_FLOAT) or
            (self.get( "nnr_compressed_data_unit_payload_type" ) == CompressedDataUnitPayloadType.NNR_PT_BLOCK)
        ):
             self.process("codebook_present_flag", "u", 1)
             if self.get( "codebook_present_flag" ):
                 self.codebook("")
        
        if (
            (self.get("nnr_compressed_data_unit_payload_type") == CompressedDataUnitPayloadType.NNR_PT_INT) or
            (self.get("nnr_compressed_data_unit_payload_type") == CompressedDataUnitPayloadType.NNR_PT_FLOAT) or
            (self.get("nnr_compressed_data_unit_payload_type") == CompressedDataUnitPayloadType.NNR_PT_BLOCK)
        ):
            self.process("dq_flag", "u", 1)    

        if self.get("nnr_decompressed_data_format_present_flag") == 1:
            self.process( "nnr_decompressed_data_format", "u", 7 )

        if self.get("input_parameters_present_flag") == 1:
            self.process( "tensor_dimensions_flag", "u", 1 )
            self.process( "cabac_unary_length_flag", "u", 1 )
            self.process( "compressed_parameter_types", "u", 4 )
            if self.get( "compressed_parameter_types" ) & BlockParameterTypes.NNR_CPT_DC != 0:
                self.process( "decomposition_rank", "ue", 3 )
                self.process( "g_number_of_rows", "ue", 3 )
            if self.get( "tensor_dimensions_flag" ) == 1:
                self.tensor_dimensions_list()
            if self.get("nnr_compressed_data_unit_payload_type") != CompressedDataUnitPayloadType.NNR_PT_BLOCK:
                if self.get("nnr_multiple_topology_elements_present_flag") == 1:
                    self.topology_tensor_dimension_mapping()
            if self.get( "cabac_unary_length_flag" ) == 1:
                self.process( "cabac_unary_length_minus1", "u", 8 )
        yield # pause execution so that tensor dimensions can be properly set based on compressed_parameter_types
        
        if (
            ( self.get("nnr_compressed_data_unit_payload_type") == CompressedDataUnitPayloadType.NNR_PT_BLOCK) and
            ((self.get("compressed_parameter_types") & BlockParameterTypes.NNR_CPT_DC) != 0) and
            ( self.get("codebook_present_flag") )
        ):
                self.codebook("dc")

        if self.get("count_tensor_dimensions") > 1:
                self.process( "scan_order", "u", 4 )
                if( self.get("scan_order") > 0 ):
                    tensorDimensions   = self.get( "tensor_dimensions" )
                    blockDim           = 4 << self.get("scan_order")
                    
                    if self.get( "compressed_parameter_types" ) & BlockParameterTypes.NNR_CPT_DC != 0:
                        hNumberOfColumns  = np.int32(np.prod( tensorDimensions )/self.get("g_number_of_rows"))
                        tensorDimensionsG = [self.get("g_number_of_rows"), self.get("decomposition_rank")] 
                        tensorDimensionsH = [self.get("decomposition_rank"), hNumberOfColumns]

                    if (
                        (self.get("nnr_compressed_data_unit_payload_type") != CompressedDataUnitPayloadType.NNR_PT_BLOCK) and
                        (self.get( "compressed_parameter_types" ) & BlockParameterTypes.NNR_CPT_DC != 0)
                    ):
                        if self.get("_decomposed_tensor_type") == "G":
                            tensorDimensions = tensorDimensionsG
                        else:
                            tensorDimensions = tensorDimensionsH

                    numBlockRowsMinus1 = ((tensorDimensions[0]+blockDim-1) >> (2+self.get("scan_order"))) - 1

                    if (
                        (self.get("nnr_compressed_data_unit_payload_type") == CompressedDataUnitPayloadType.NNR_PT_BLOCK) and
                        (self.get( "compressed_parameter_types" ) & BlockParameterTypes.NNR_CPT_DC != 0)
                    ):
                        numBlockRowsMinus1  = ((tensorDimensionsG[0]+blockDim-1) >> (2+self.get("scan_order"))) - 1
                        numBlockRowsMinus1 += ((tensorDimensionsH[0]+blockDim-1) >> (2+self.get("scan_order"))) - 1

                    self.process("cabac_entry_point_list", "entry_point_list", numBlockRowsMinus1, self.get("dq_flag"))            

        self.__coder.byte_alignment()
        
    def tensor_dimensions_list(self):
        self.process( "count_tensor_dimensions", "ue", 1 )
        self.define_array( "tensor_dimensions", [self.get( "count_tensor_dimensions" )], np.uint32 )
        for j in range( self.get( "count_tensor_dimensions" ) ):
            self.process( "tensor_dimensions[%d]" % j, "ue", 7 )

    def topology_elements_ids_list(self, topologyIndexedFlag): 
        self.process( "count_topology_elements_minus2", "ue", 7 )
        if topologyIndexedFlag == 0:
            self.__coder.byte_alignment()
            self.define_array( "topology_elem_id_list", [self.get( "count_topology_elements_minus2" ) + 2], str )
        else:
            self.define_array( "topology_elem_id_index_list", [self.get( "count_topology_elements_minus2" ) + 2], np.uint32 )
        for j in range( self.get( "count_topology_elements_minus2" ) + 2 ):
            if topologyIndexedFlag == 0:
                self.process( "topology_elem_id_list[%d]" % j, "st" )
            else:
               self.process( "topology_elem_id_index_list[%d]" % j, "ue", 7 )
        if topologyIndexedFlag == 1:
           self.__coder.byte_alignment()
    
    def topology_tensor_dimension_mapping(self):
        raise NotImplementedError("topology_tensor_dimension_mapping not yet implemented!")

    def nnr_aggregate_unit_header(self):
        raise NotImplementedError("nnr_aggregate_unit_header not yet implemented!")
    
    def nnr_unit_payload(self):
        if self.get( "nnr_unit_type" ) == NnrUnitType.NNR_MPS:
            self.nnr_model_parameter_set_unit_payload()
        if self.get( "nnr_unit_type" ) == NnrUnitType.NNR_LPS:
            self.nnr_layer_parameter_set_unit_payload()
        if self.get( "nnr_unit_type" ) == NnrUnitType.NNR_TPL:
            self.nnr_topology_unit_payload()
        if self.get( "nnr_unit_type" ) == NnrUnitType.NNR_QNT:
            self.nnr_quanization_unit_payload()
        if self.get( "nnr_unit_type" ) == NnrUnitType.NNR_NDU:
            self.nnr_compressed_data_unit_payload()
        if self.get( "nnr_unit_type" ) == NnrUnitType.NNR_STR:
            self.nnr_start_unit_payload()
        if self.get( "nnr_unit_type" ) == NnrUnitType.NNR_AGG:
            self.nnr_aggregate_unit_payload()

    def nnr_start_unit_payload(self):
        #nnr_reserved_zero_0bit, "u", 0
        pass

    def nnr_model_parameter_set_unit_payload(self):
        self.process( "topology_carriage_flag",    "u",  1 )
        self.process( "mps_sparsification_flag",   "u",  1 )
        self.process( "mps_pruning_flag",          "u",  1 )
        self.process( "mps_unification_flag",      "u",  1 )
        self.process( "mps_decomposition_performance_map_flag", "u",  1 )
        self.process( "mps_quantization_method_flags", "u",  3 )
        self.process( "mps_topology_indexed_reference_flag", "u", 1 )
        self.process( "nnr_reserved_zero_7bits", "u", 7 )
        if (
            (self.get( "mps_quantization_method_flags" ) & QuantizationMethodFlags.NNR_QSU != 0) or
            (self.get( "mps_quantization_method_flags" ) & QuantizationMethodFlags.NNR_QCB != 0)
        ):
            self.process( "mps_qp_density",             "u",  3 )
            self.process( "mps_quantization_parameter", "i", 13 )
        if self.get( "mps_sparsification_flag" ) == 1:
            self.sparsification_performance_map()
        if self.get( "mps_pruning_flag" ) == 1:
            self.pruning_performance_map()
        if self.get( "mps_unification_flag" ) == 1:
            self.unification_performance_map()
        if self.get( "mps_decomposition_performance_map_flag" ) == 1:
            self.decomposition_performance_map()
        self.__coder.byte_alignment()

    def sparsification_performance_map(self):
        self.process( "spm_count_thresholds", "u", 8 )
        spm_count_threshold_minus1 = self.get("spm_count_thresholds") -1
        self.define_array( "sparsification_threshold", [spm_count_threshold_minus1]    , np.float32 )
        self.define_array( "non_zero_ratio"          , [spm_count_threshold_minus1]    , np.float32 )
        self.define_array( "spm_nn_accuracy"         , [spm_count_threshold_minus1]    , np.float32 )
        self.define_array( "spm_count_classes"       , [spm_count_threshold_minus1]    , np.uint32  )
        self.define_array( "spm_class_bitmask"       , [spm_count_threshold_minus1]    , np.uint32  )
        self.define_array( "spm_nn_class_accuracy"   , [spm_count_threshold_minus1, 1] , np.float32 )
        for i in range( spm_count_threshold_minus1 ):
            self.process( "sparsification_threshold[%d]" % i, "flt", 32 )
            self.process( "non_zero_ratio[%d]" % i          , "flt", 32 )
            self.process( "spm_nn_accuracy[%d]" % i         , "flt", 32 )
            self.process( "spm_count_classes[%d]" % i       , "u"  ,  8 )
            self.process( "spm_class_bitmask[%d]" % i       , "ue" ,  7 )
            self.extend_second_2darray_dimension( "spm_nn_class_accuracy", np.max(self.get("spm_count_classes")), np.float32)
            for j in range( self.get("spm_count_classes")[i] ):
                self.process("spm_nn_class_accuracy[%d][%d]" % (i,j), "flt", 32)

    def pruning_performance_map(self):
        self.process( "ppm_count_pruning_ratios", "u", 8 )
        ppm_count_pruning_ratios_minus1 = self.get("ppm_count_pruning_ratios") -1
        self.define_array( "pruning_ratio"        , [ppm_count_pruning_ratios_minus1]    , np.float32 )
        self.define_array( "ppm_nn_accuracy"      , [ppm_count_pruning_ratios_minus1]    , np.float32 )
        self.define_array( "ppm_count_classes"    , [ppm_count_pruning_ratios_minus1]    , np.uint32 )
        self.define_array( "ppm_class_bitmask"    , [ppm_count_pruning_ratios_minus1]    , np.uint32 )
        self.define_array( "ppm_nn_class_accuracy", [ppm_count_pruning_ratios_minus1, 1] , np.float32 )
        for i in range( ppm_count_pruning_ratios_minus1 ):
            self.process("pruning_ratio[%d]" % i    , "flt", 32)
            self.process("ppm_nn_accuracy[%d]" % i  , "flt", 32)
            self.process("ppm_count_classes[%d]" % i, "u"  ,  8)
            self.process("ppm_class_bitmask[%d]" % i, "ue" ,  7)
            self.extend_second_2darray_dimension( "ppm_nn_class_accuracy", np.max(self.get("ppm_count_classes")), np.float32)
            for j in range( self.get("ppm_count_classes")[i] ):
                self.process("ppm_nn_class_accuracy[%d][%d]" % (i,j), "flt", 32)

    def unification_performance_map(self):
        self.process("upm_count_thresholds", "u", 8)
        upm_count_thresholds_minus1 = self.get("upm_count_thresholds") -1
        self.define_array( "count_reshaped_tensor_dimension" , [upm_count_thresholds_minus1]    , np.uint32  )
        self.define_array( "reshaped_tensor_dimensions"      , [upm_count_thresholds_minus1, 1] , np.uint32  )
        self.define_array( "count_super_block_dimension"     , [upm_count_thresholds_minus1]    , np.uint32  )
        self.define_array( "super_block_dimensions"          , [upm_count_thresholds_minus1, 1] , np.uint32  )
        self.define_array( "count_block_dimension"           , [upm_count_thresholds_minus1]    , np.uint32  )
        self.define_array( "block_dimensions"                , [upm_count_thresholds_minus1, 1] , np.uint32  )
        self.define_array( "unification_threshold"           , [upm_count_thresholds_minus1]    , np.float32 )
        self.define_array( "upm_nn_accuracy"                 , [upm_count_thresholds_minus1]    , np.float32 )
        self.define_array( "upm_count_classes"               , [upm_count_thresholds_minus1]    , np.uint32  )
        self.define_array( "upm_class_bitmask"               , [upm_count_thresholds_minus1]    , np.uint32  )
        self.define_array( "upm_nn_class_accuracy"           , [upm_count_thresholds_minus1, 1] , np.float32 )
        for i in range( upm_count_thresholds_minus1 ):
            self.process( "count_reshaped_tensor_dimension[%d]" % i, "ue", 1)
            self.extend_second_2darray_dimension( "reshaped_tensor_dimensions", np.max(self.get("count_reshaped_tensor_dimension")), np.uint32)
            for j in range( self.get( "count_reshaped_tensor_dimension" )[i] - 1 ):
                self.process( "reshaped_tensor_dimensions[%d][%d]" % (i,j), "ue", 7 )
            self.__coder.byte_alignment()
            self.process( "count_super_block_dimension[%d]" % i, "u", 8)
            self.extend_second_2darray_dimension( "super_block_dimensions", np.max(self.get("count_super_block_dimension")), np.uint32)
            for j in range( self.get( "count_super_block_dimension" )[i] - 1 ):
                self.process( "super_block_dimensions[%d][%d]" % (i,j), "u", 8 )
            self.process( "count_block_dimension[%d]" % i      , "u", 8)
            self.extend_second_2darray_dimension( "block_dimensions", np.max(self.get("count_block_dimension")), np.uint32)
            for j in range( self.get( "count_block_dimension" )[i] - 1 ):
                self.process( "block_dimensions[%d][%d]" % (i,j)      , "u", 8 )
            self.process( "unification_threshold[%d]" % i , "flt", 32 )
            self.process( "upm_nn_accuracy[%d]" % i       , "flt", 32 )
            self.process( "upm_count_classes[%d]" % i     , "u"  ,  8 )
            self.process( "upm_class_bitmask[%d]" % i     , "ue" ,  7 )
            self.extend_second_2darray_dimension( "upm_nn_class_accuracy", np.max(self.get("upm_count_classes")), np.float32)
            for j in range( self.get( "upm_class_bitmask" )[i] ):
                self.process( "upm_nn_class_accuracy[%d][%d]" % (i,j), "flt", 32 )

    def decomposition_performance_map(self):
        self.process("dpm_count_thresholds", "u", 8)
        dpm_count_thresholds_minus1 = self.get("dpm_count_thresholds") - 1
        self.define_array( "mse_threshold"         , [dpm_count_thresholds_minus1]    , np.float32  )
        self.define_array( "dpm_nn_accuracy"       , [dpm_count_thresholds_minus1]    , np.float32  )
        self.define_array( "nn_reduction_ratio"    , [dpm_count_thresholds_minus1]    , np.float32  )
        self.define_array( "dpm_count_classes"     , [dpm_count_thresholds_minus1]    , np.uint32  )
        self.define_array( "dpm_nn_class_accuracy" , [dpm_count_thresholds_minus1, 1] , np.float32  )
        for i in range( dpm_count_thresholds_minus1 ):
            self.process( "mse_threshold[%d]" % i      , "flt", 32 )
            self.process( "dpm_nn_accuracy[%d]" % i    , "flt", 32 )
            self.process( "nn_reduction_ratio[%d]" % i , "flt", 32 )
            self.process( "dpm_count_classes[%d]" % i  , "u"  , 16 )
            self.extend_second_2darray_dimension( "dpm_nn_class_accuracy", np.max(self.get("dpm_count_classes")), np.float32)
            for j in range( self.get( "dpm_count_classes" )[i] ):
                self.process( "dpm_nn_class_accuracy[%d][%d]" % (i,j), "flt", 32 )
    
    def nnr_layer_parameter_set_unit_payload(self):
        self.process( "nnr_reserved_zero_1_bits"      , "u", 1 )
        self.process( "lps_sparsification_flag"       , "u", 1 )
        self.process( "lps_pruning_flag"              , "u", 1 )
        self.process( "lps_unification_flag"          , "u", 1 )
        self.process( "lps_quantization_method_flags" , "u", 3 )
        self.process( "nnr_reserved_zero_1bit"        , "u", 1 )
        if(
            (self.get( "lps_quantization_method_flags" ) & QuantizationMethodFlags.NNR_QCB != 0) or
            (self.get( "lps_quantization_method_flags" ) & QuantizationMethodFlags.NNR_QSU != 0)
        ):
            self.process( "lps_qp_density"            , "u", 3 )
            self.process( "lps_quantization_parameter", "i", 13 )
        if self.get("lps_sparsification_flag") == 1:
            self.sparsification_performance_map()
        if self.get("lps_pruning_flag")        == 1:
            self.pruning_performance_map()
        if self.get("lps_unification_flag")    == 1:
            self.unification_performance_map()
        self.__coder.byte_alignment()

    def nnr_topology_unit_payload(self):
        if self.get( "topology_storage_format" ) == nnr_model.TopologyStorageFormat.NNR_TPL_PRUN:
            self.nnr_pruning_topology_container()
        elif self.get( "topology_storage_format" ) == nnr_model.TopologyStorageFormat.NNR_TPL_REFLIST:
            self.topology_elements_ids_list(0)
        else:
            self.process( "topology_data", "st" )##tbd: implement bs(v) 
    
    def nnr_pruning_topology_container(self):
        raise NotImplementedError("Pruning Topology Container not yet implemented!")

    def nnr_quanization_unit_payload(self):
        self.process( "quantization_data", "st" )

    def nnr_compressed_data_unit_payload(self):
        if self.get( "nnr_compressed_data_unit_payload_type" ) == CompressedDataUnitPayloadType.NNR_PT_RAW_FLOAT:
            self.process( "raw_float32_parameter", "flt_tensor", 32, self.get( "tensor_dimensions" ) )

    def nnr_aggregate_unit_payload(self):
        pass
    
def encode_nnr_unit_with_size_dummy(nnr_unit):
    bs = bytearray()
    w = BitWriter( bs )
    w.u( 32, 0 ) # reserve 4 bytes for nnr_unit_size()
    hls_enc = Coder( w, nnr_unit )
    for _ in hls_enc.nnr_unit_header(): pass
    hls_enc.nnr_unit_payload()
    return bs


def decode_nnr_unit(bitstream, nnr_unit):
    reader = BitReader(bitstream)
    yield from decode_nnr_unit_size_and_header(reader, nnr_unit)
    decode_nnr_unit_payload(reader, nnr_unit)
    yield reader.getNumBytesTouched()

    
def decode_nnr_unit_size_and_header(bit_reader, nnr_unit):
    hls_dec = Coder( bit_reader, nnr_unit )
    hls_dec.nnr_unit_size()
    yield from hls_dec.nnr_unit_header()
    yield

def decode_nnr_unit_payload( bit_reader, nnr_unit_size_and_header ):
    hls_dec = Coder( bit_reader, nnr_unit_size_and_header )
    hls_dec.nnr_unit_payload()


def update_nnr_unit_size( bs ):
    unit_size = {}
    unit_size["nnr_unit_size_flag"] = 1
    if len( bs ) - 2 < 32768:
        # short nnr_unit_size
        bs = bs[2:]
        unit_size["nnr_unit_size_flag"] = 0

    unit_size["nnr_unit_size"] = len( bs )
    w = BitWriter( bs )
    hls_enc = Coder( w, unit_size )
    hls_enc.nnr_unit_size()
    return bs, unit_size

/*
 * Copyright (c) 2023-2024 The ggml authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
#include "ggml-impl.h"
#include "ggml-common.h"
#include "ggml-qnn-ops.h"

#define GGMLQNN_CHECK_PARAMS(ctx, src0, src1, dst)                          \
    do {                                                                    \
        if (!ggmlqnn_is_valid_params((ctx), (src0), (src1), (dst))) {       \
            return;                                                         \
        }                                                                   \
    } while (0)

/*
 * provide a general skeleton to offload ggml op to QNN backend: a single node contains 2 input
 * tensor and 1 output tensor
*/
void ggml_qnn_general_node(ggml_backend_qnn_context * ctx, ggml_tensor * op) {
    Qnn_ErrorHandle_t error                     = QNN_SUCCESS;
    enum ggml_status result                     = GGML_STATUS_SUCCESS;
    bool graph_initialized                      = false;
    qnn_instance * instance                     = nullptr;
    Qnn_GraphHandle_t graph_handle              = nullptr;
    Qnn_Tensor_t * p_tensor0                    = nullptr;
    Qnn_Tensor_t * p_tensor1                    = nullptr;
    Qnn_Tensor_t * p_tensor2                    = nullptr;
    Qnn_Param_t qnn_params[]                    = {};
    const ggml_tensor * src0                    = op->src[0];
    const ggml_tensor * src1                    = op->src[1];
    ggml_tensor * dst                           = op;

    GGMLQNN_CHECK_PARAMS(ctx, src0, src1, dst);
    instance                                    = ctx->instance;
    QNN_INTERFACE_VER_TYPE qnn_raw_interface    = ctx->raw_interface;
    size_t qnn_op_index                         = ggmlqnn_get_op_index(op);
    GGML_ASSERT(qnn_op_index < ggmlqnn_get_opcaps_size());
    const char * qnn_op_name                    = k_op_caps[qnn_op_index].qnn_op_name;
    std::string ggml_op_name_string             = std::string("ggml_") + ggml_op_name(op->op);
    const char * ggml_op_name                   = ggml_op_name_string.c_str();

    qnn_perf op_perf                            = qnn_perf(ggml_op_name);
    op_perf.start();

    std::string graph_name;
    ggmlqnn_get_graphkey_from_op(op, graph_name);
    if (instance->_qnn_graph_map.find(graph_name) != instance->_qnn_graph_map.end()) {
        graph_initialized = true;
        qnn_res_t & graph_item = instance->_qnn_graph_map[graph_name];
        graph_handle = std::get<0>(graph_item);
        qnn_tensors_t & tensor = std::get<1>(graph_item);
        p_tensor0     = tensor[0];
        p_tensor1     = tensor[1];
        p_tensor2     = tensor[2];
    } else {
        p_tensor0 = ggmlqnn_create_compute_tensor(src0);
        p_tensor1 = ggmlqnn_create_compute_tensor(src1);
        p_tensor2 = ggmlqnn_create_compute_tensor(dst);
    }
#if GGMLQNN_PRINT_OP_ADD_LOG
    print_tensors_info(__func__, ctx, src0, src1, dst);
#endif

    //ensure QNN tensor has correct tensor type
    QNN_VER_PTR(*p_tensor0)->type = QNN_TENSOR_TYPE_APP_WRITE;
    QNN_VER_PTR(*p_tensor1)->type = QNN_TENSOR_TYPE_APP_WRITE;
    QNN_VER_PTR(*p_tensor2)->type = QNN_TENSOR_TYPE_APP_READ;

    //save the original dimensions of qnn tensors
    uint32_t * tensor_0_dimensions = QNN_VER_PTR(*p_tensor0)->dimensions;
    uint32_t * tensor_1_dimensions = QNN_VER_PTR(*p_tensor1)->dimensions;
    uint32_t * tensor_2_dimensions = QNN_VER_PTR(*p_tensor2)->dimensions;

    bool enable_npu_rpc = instance->enable_qnn_rpc() && ctx->device == QNN_BACKEND_NPU;

    if (!graph_initialized) {
        GGMLQNN_LOG_DEBUG("graph name %s", graph_name.c_str());
        error = instance->init_qnn_graph(graph_name, static_cast<QNNBackend>(ctx->device), 8);
        if (QNN_SUCCESS != error) {
            GGMLQNN_LOG_INFO("can't create qnn graph handle with graph name %s, error = %d\n", graph_name.c_str(), error);
            return;
        }
        graph_handle = instance->get_qnn_graph_handle();

        if (enable_npu_rpc) {
            QNN_VER_PTR(*p_tensor0)->memType = QNN_TENSORMEMTYPE_MEMHANDLE;
            QNN_VER_PTR(*p_tensor0)->clientBuf = {.data=nullptr, .dataSize=0};

            QNN_VER_PTR(*p_tensor1)->memType = QNN_TENSORMEMTYPE_MEMHANDLE;
            QNN_VER_PTR(*p_tensor1)->clientBuf = {.data=nullptr, .dataSize=0};

            QNN_VER_PTR(*p_tensor2)->memType = QNN_TENSORMEMTYPE_MEMHANDLE;
            QNN_VER_PTR(*p_tensor2)->clientBuf = {.data=nullptr, .dataSize=0};
        }

        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_tensor0));
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_tensor1));
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_tensor2));

        if (enable_npu_rpc) {
            uint8_t * qnn_rpcbuffer_0 = ggmlqnn_create_rpc_buffer(instance, src0, p_tensor0, true);
            uint8_t * qnn_rpcbuffer_1 = ggmlqnn_create_rpc_buffer(instance, src1, p_tensor1, true);
            uint8_t * qnn_rpcbuffer_2 = ggmlqnn_create_rpc_buffer(instance, dst, p_tensor2, false);
            if (nullptr == qnn_rpcbuffer_0 || nullptr == qnn_rpcbuffer_1 || nullptr == qnn_rpcbuffer_2) {
                GGMLQNN_LOG_INFO("create rpc buffer failure\n");
                //TODO: potential memory leak although it shouldn't happen
                return;
            }
        } else {
            QNN_VER_PTR(*p_tensor0)->clientBuf = {src0->data, static_cast<uint32_t>(ggml_nbytes(src0))};
            QNN_VER_PTR(*p_tensor1)->clientBuf = {src1->data, static_cast<uint32_t>(ggml_nbytes(src1))};
            QNN_VER_PTR(*p_tensor2)->clientBuf = {dst->data, static_cast<uint32_t>(ggml_nbytes(dst))};
        }

        Qnn_Tensor_t tensor_inputs[] = {
                *p_tensor0,
                *p_tensor1
        };
        Qnn_Tensor_t tensor_outputs[] = {
                *p_tensor2
        };
        Qnn_OpConfig_t op_config = {
                QNN_OPCONFIG_VERSION_1, .v1 = {
                        ggml_op_name,
                        QNN_OP_PACKAGE_NAME_QTI_AISW,
                        qnn_op_name,
                        0,
                        qnn_params,
                        2,
                        tensor_inputs,
                        1,
                        tensor_outputs
                }
        };
        CHECK_QNN_API(error, qnn_raw_interface.graphAddNode(graph_handle, op_config));
        CHECK_QNN_API(error, qnn_raw_interface.graphFinalize(graph_handle, nullptr, nullptr));
        CHECK_QNN_API(error, qnn_raw_interface.graphExecute(graph_handle,
                                                            tensor_inputs, 2,
                                                            tensor_outputs, 1,
                                                            nullptr, nullptr));

        if (enable_npu_rpc) {
            uint8_t * qnn_rpcbuffer = static_cast<uint8_t *>(instance->get_rpcmem_from_memhandle(QNN_VER_PTR(*p_tensor2)->memHandle));
            GGMLQNN_LOG_INFO("qnn_rpcbuffer = %p\n", qnn_rpcbuffer);
            if (nullptr != qnn_rpcbuffer) {
                memcpy(dst->data, qnn_rpcbuffer, ggml_nbytes(dst));
            }
        }

        qnn_tensors_t ggml_op_add_tensors;
        ggml_op_add_tensors.reserve(3);
        ggml_op_add_tensors.push_back(p_tensor0);
        ggml_op_add_tensors.push_back(p_tensor1);
        ggml_op_add_tensors.push_back(p_tensor2);

        auto  graph_item = std::make_tuple(graph_handle, ggml_op_add_tensors);
        instance->_qnn_graph_map[graph_name] = graph_item;
    } else {
        Qnn_DataType_t src0_qnn_type    = QNN_DATATYPE_FLOAT_32;
        Qnn_DataType_t src1_qnn_type    = QNN_DATATYPE_FLOAT_32;
        Qnn_DataType_t dst_qnn_type     = QNN_DATATYPE_FLOAT_32;

        src0_qnn_type                   = ggmlqnn_datatype_from_ggml_datatype(src0->type);
        src1_qnn_type                   = ggmlqnn_datatype_from_ggml_datatype(src1->type);
        dst_qnn_type                    = ggmlqnn_datatype_from_ggml_datatype(dst->type);

        uint32_t dimensions_input_0[] = {(uint32_t) src0->ne[0], (uint32_t) src0->ne[1],
                                         (uint32_t) src0->ne[2], (uint32_t) src0->ne[3]};
        uint32_t dimensions_input_1[] = {(uint32_t) src1->ne[0], (uint32_t) src1->ne[1],
                                         (uint32_t) src1->ne[2], (uint32_t) src1->ne[3]};
        uint32_t dimensions_output[]  = {(uint32_t) dst->ne[0], (uint32_t) dst->ne[1],
                                         (uint32_t) dst->ne[2], (uint32_t) dst->ne[3]};

        QNN_VER_PTR(*p_tensor0)->dimensions  = dimensions_input_0;
        QNN_VER_PTR(*p_tensor0)->rank        = ggml_n_dims(src0);
        QNN_VER_PTR(*p_tensor0)->dataType    = src0_qnn_type;

        QNN_VER_PTR(*p_tensor1)->dimensions  = dimensions_input_1;
        QNN_VER_PTR(*p_tensor1)->rank        = ggml_n_dims(src1);
        QNN_VER_PTR(*p_tensor1)->dataType    = src1_qnn_type;

        QNN_VER_PTR(*p_tensor2)->dimensions  = dimensions_output;
        QNN_VER_PTR(*p_tensor2)->rank        = ggml_n_dims(dst);
        QNN_VER_PTR(*p_tensor2)->dataType    = dst_qnn_type;

        if (enable_npu_rpc) {
            //TODO: NPU RPC feature will failed with test-backend-ops
            uint8_t * qnn_buffer_0 = static_cast<uint8_t *>(instance->get_rpcmem_from_memhandle(QNN_VER_PTR(*p_tensor0)->memHandle));
            GGMLQNN_LOG_INFO("qnn_rpcbuffer_0 = %p\n", qnn_buffer_0);
            if (nullptr != qnn_buffer_0) {
                memcpy(qnn_buffer_0, src0->data, ggml_nbytes(src0));
            }

            uint8_t * qnn_buffer_1 = static_cast<uint8_t *>(instance->get_rpcmem_from_memhandle(QNN_VER_PTR(*p_tensor1)->memHandle));
            GGMLQNN_LOG_INFO("qnn_rpcbuffer_1 = %p\n", qnn_buffer_1);
            if (nullptr != qnn_buffer_1) {
                memcpy(qnn_buffer_1, src1->data, ggml_nbytes(src1));
            }
        } else {
            QNN_VER_PTR(*p_tensor0)->clientBuf = {src0->data, static_cast<uint32_t>(ggml_nbytes(src0))};
            QNN_VER_PTR(*p_tensor1)->clientBuf = {src1->data, static_cast<uint32_t>(ggml_nbytes(src1))};
            QNN_VER_PTR(*p_tensor2)->clientBuf = {dst->data, static_cast<uint32_t>(ggml_nbytes(dst))};
        }

        Qnn_Tensor_t tensor_inputs[] = {
                *p_tensor0,
                *p_tensor1
        };
        Qnn_Tensor_t tensor_outputs[] = {
                *p_tensor2
        };
        CHECK_QNN_API(error, qnn_raw_interface.graphExecute(graph_handle,
                                                            tensor_inputs, 2,
                                                            tensor_outputs, 1,
                                                            nullptr, nullptr));

        if (enable_npu_rpc) {
            //TODO:NPU RPC feature will failed with test-backend-ops
            uint8_t * qnn_buffer_2 = static_cast<uint8_t *>(instance->get_rpcmem_from_memhandle(QNN_VER_PTR(*p_tensor2)->memHandle));
            if (nullptr != qnn_buffer_2) {
                memcpy(dst->data, qnn_buffer_2, ggml_nbytes(dst));
            }
        }
    }

    // restore the original dimensions of qnn tensors to avoid memory leak in func free_qnn_tensor
    QNN_VER_PTR(*p_tensor0)->dimensions = tensor_0_dimensions;
    QNN_VER_PTR(*p_tensor1)->dimensions = tensor_1_dimensions;
    QNN_VER_PTR(*p_tensor2)->dimensions = tensor_2_dimensions;

#if GGMLQNN_PRINT_OP_ADD_LOG
    op_perf.info();
#endif
}

/*
 * @brief performs matrix multiplication with FP32 & quantized weights and floating-point inputs
 *        using the QNN backend. this function performs matrix multiplication of the input tensor
 *        `src1` and the weight tensor `src0`, handling transposing, and quantization as needed,
 *        and stores the result in the destination tensor `dst`.
 *
 * @param backend the context which got through (ggml_backend_qnn_context *)backend->context for the
 *                QNN backend operations.
 * @param op      the destination tensor where the result of the matrix multiplication will be stored.
 *
 * @note the logic of ggml_qnn_mul_mat is similar to ggml_qnn_general_node but much more complicated
 *       than ggml_qnn_general_node. so it's a standalone function. accordingly, this is another
 *       typical skeleton for offload other ggml ops to QNN backend. MUL_MAT take most of the compute
 *       time (about 95%).so to speed up llama inference, should focus on this func. there are three kinds
 *       of MUL_MAT to compute:
 *       mul_mat_f32:     both src0 and src1 are F32, this will be naturally handled in QNN backend
 *       mul_mat_f16_f32: src0 is F16 and src1 is F32, f16 in src0 -> f32 in src0', then src0' * src1
 *       mul_mat_q_f32:   src0 is quantized (Q4_0, Q4_1, Q6_K...)
 *                        and src1 is F32, src0 -> f32 in src0', then src0' * src1
*/
void ggml_qnn_mul_mat(ggml_backend_qnn_context * ctx, ggml_tensor * op) {
    Qnn_ErrorHandle_t error                     = QNN_SUCCESS;
    bool graph_initialized                      = false;
    qnn_perf op_perf                            = qnn_perf("ggml_qnn_mul_mat");
    qnn_instance * instance                     = nullptr;
    Qnn_GraphHandle_t graph_handle              = nullptr;
    Qnn_Tensor_t * p_tensor0                    = nullptr;
    Qnn_Tensor_t * p_tensor1                    = nullptr;
    Qnn_Tensor_t * p_tensor2                    = nullptr;
    Qnn_Tensor_t * p_param_tensor               = nullptr;
    Qnn_Tensor_t * p_tensor2_transpose          = nullptr;
    const ggml_tensor * src0                    = op->src[0];
    const ggml_tensor * src1                    = op->src[1];
    ggml_tensor       * dst                     = op;

    GGMLQNN_CHECK_PARAMS(ctx, src0, src1, dst);
    instance                                    = ctx->instance;
    QNN_INTERFACE_VER_TYPE qnn_raw_interface    = ctx->raw_interface;
    op_perf.start();

    const enum ggml_type src0_type              = src0->type;
    const uint32_t src0_rank                    = ggml_n_dims(src0);
    const uint32_t src1_rank                    = ggml_n_dims(src1);
    GGML_ASSERT(src0_rank == src1_rank);
    GGML_ASSERT(src0_rank >= 2); //QNN SDK's limitation, make QNN SDK happy
    GGML_ASSERT(src0_rank != 4); //TODO: 4D matrix mulmat
    void * wdata                                = ggmlqnn_type_trait(ctx, op);
    const size_t desired_size                   = ctx->desired_size;

    std::string graph_name;
    ggmlqnn_get_graphkey_from_op(op, graph_name);
    if (instance->_qnn_graph_map.find(graph_name) != instance->_qnn_graph_map.end()) {
        graph_initialized       = true;
        qnn_res_t & graph_item  = instance->_qnn_graph_map[graph_name];
        graph_handle            = std::get<0>(graph_item);
        qnn_tensors_t & tensors = std::get<1>(graph_item);
        p_tensor0               = tensors[0];
        p_tensor1               = tensors[1];
        p_tensor2               = tensors[2];
        p_param_tensor          = tensors[3];
        p_tensor2_transpose     = tensors[4];
    } else {
        p_tensor0 = GQCGT(src0, nullptr, QNN_TENSOR_TYPE_APP_WRITE,QNN_DATATYPE_FLOAT_32, src0_rank, nullptr, nullptr, 0);
        p_tensor1 = GQCGT(src1, nullptr, QNN_TENSOR_TYPE_APP_WRITE,QNN_DATATYPE_FLOAT_32, src0_rank, nullptr, nullptr, 0);
        p_tensor2 = GQCGT(dst, nullptr, QNN_TENSOR_TYPE_APP_READ,QNN_DATATYPE_FLOAT_32, src0_rank, nullptr, nullptr, 0);
    }
    ggmlqnn_print_tensors_info(__func__, ctx, src0, src1, dst);

    //ensure QNN tensor has correct tensor type
    QNN_VER_PTR(*p_tensor0)->type = QNN_TENSOR_TYPE_APP_WRITE;
    QNN_VER_PTR(*p_tensor1)->type = QNN_TENSOR_TYPE_APP_WRITE;
    QNN_VER_PTR(*p_tensor2)->type = QNN_TENSOR_TYPE_APP_READ;

    //save the original dimensions of qnn tensors
    uint32_t * tensor_0_dimensions = QNN_VER_PTR(*p_tensor0)->dimensions;
    uint32_t * tensor_1_dimensions = QNN_VER_PTR(*p_tensor1)->dimensions;
    uint32_t * tensor_2_dimensions = QNN_VER_PTR(*p_tensor2)->dimensions;

    if (!graph_initialized) {
        GGMLQNN_LOG_DEBUG("graph name %s", graph_name.c_str());
        /*
         there are two key-points in properly handling how to offload mulmat to the QNN backend in ggml-qnn
         1. transpose
            a 3x2 f32 matrix which means 3 rows and 2 columns. in ggml, it could be created from:
            struct ggml_tensor* matrix = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 3);
            which like this:
            +---+---+
            | 0 | 1 |
            +---+---+
            | 2 | 3 |
            +---+---+
            | 4 | 5 |
            +---+---+
            with
                ne[0] = 2
                ne[1] = 3
            there are different dimension order between ggml tensor and qnn tensor

          2. QNN's MatMul can only support input tensors with rank >= 2

             in the all, there is gap between ggml mulmat and QNN mulmat,we need to perform a transpose
             operation when offloading mulmat to QNN backend. this concise implementation will handle
             transpose in func ggml_qnn_create_general_tensor()
        */
        //step-1: create qnn graph
        error = qnn_raw_interface.graphCreate(instance->get_qnn_context_handle(),
                                              graph_name.c_str(), nullptr, &graph_handle);
        if (QNN_SUCCESS != error) {
            GGMLQNN_LOG_INFO("can't create qnn graph handle with graph name %s, error = %d\n", graph_name.c_str(), error);
            return;
        }
        //step-2: create param tensor for mulmat of 2d/3d/4d matrix
        const uint32_t param_tensor_data[GGML_MAX_DIMS][GGML_MAX_DIMS] = {
                {0},
                {1, 0},
                {0, 2, 1},
                {0, 1, 3, 2},
        };
        uint32_t param_tensor_dims[1]   = {src0_rank};
        p_param_tensor = GQCGT(nullptr, "param", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_UINT_32, 1, param_tensor_dims, (void *)(param_tensor_data[src0_rank - 1]), src0_rank * sizeof(uint32_t));
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_param_tensor));

        //step-3: create compute tensor from ggml tensor
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_tensor0));
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_tensor1));
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_tensor2));
        if (src0_type != GGML_TYPE_F32) {
            QNN_VER_PTR(*p_tensor0)->clientBuf = {wdata, static_cast<uint32_t>(desired_size)};
        } else {
            QNN_VER_PTR(*p_tensor0)->clientBuf = {src0->data, static_cast<uint32_t>(ggml_nbytes(src0))};
        }
        QNN_VER_PTR(*p_tensor1)->clientBuf = {src1->data, static_cast<uint32_t>(ggml_nbytes(src1))};
        QNN_VER_PTR(*p_tensor2)->clientBuf = {dst->data, static_cast<uint32_t>(ggml_nbytes(dst))};

        //step-4: create a transpose tensor
        p_tensor2_transpose = GQCGT(dst, "transpose", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32, src0_rank, nullptr, nullptr, 0, true);
        CHECK_QNN_API(error, qnn_raw_interface.tensorCreateGraphTensor(graph_handle, p_tensor2_transpose));

        //step-5: compose qnn graph: add mat_mul node
        Qnn_Param_t out_0_params[] = {
                {QNN_PARAMTYPE_SCALAR,
                 QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1,
                        .scalarParam = {QNN_DATATYPE_BOOL_8, .bool8Value = 1}
                }
        };

        Qnn_Tensor_t out_0_inputs[]  = {*p_tensor0, *p_tensor1};
        Qnn_Tensor_t out_0_outputs[] = {*p_tensor2_transpose};
#if 0 //leave here for easily understand code, can be removed in the future
        Qnn_OpConfig_t out_0 = {
                QNN_OPCONFIG_VERSION_1, .v1 =
                        {"ggmlqnn_mulmat_opconfig", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_MAT_MUL,
                         1,
                         out_0_params,
                         2,
                         out_0_inputs,
                         1,
                         out_0_outputs}
        };
#else
        Qnn_OpConfig_t out_0 = ggmlqnn_create_op_config("ggmlqnn_mulmat_opconfig", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_MAT_MUL,
                                                out_0_params, 1, out_0_inputs, 2, out_0_outputs, 1);
#endif
        CHECK_QNN_API(error, qnn_raw_interface.graphAddNode(graph_handle,out_0));

        //step-5: compose qnn graph: add transpose node
        Qnn_Param_t out_trans1_0_params[] = {
                {(Qnn_ParamType_t) 1,
                 "perm", .tensorParam = *p_param_tensor
                }
        };
        Qnn_Tensor_t out_trans1_0_inputs[]  = {*p_tensor2_transpose};
        Qnn_Tensor_t out_trans1_0_outputs[] = {*p_tensor2};
#if 0 //leave here for easily understand code, can be removed in the future
        Qnn_OpConfig_t out_trans1_0 = {
                QNN_OPCONFIG_VERSION_1,
                .v1 =  {"ggmlqnn_mulmat_transpose_opconfig",
                        QNN_OP_PACKAGE_NAME_QTI_AISW,
                        QNN_OP_TRANSPOSE, 1,
                        out_trans1_0_params,
                        1,
                        out_trans1_0_inputs,
                        1,
                        out_trans1_0_outputs}
        };
#else
        Qnn_OpConfig_t out_trans1_0 = ggmlqnn_create_op_config("ggmlqnn_mulmat_transpose_opconfig", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_TRANSPOSE,
                                                       out_trans1_0_params, 1, out_trans1_0_inputs, 1, out_trans1_0_outputs, 1);
#endif
        CHECK_QNN_API(error, qnn_raw_interface.graphAddNode(graph_handle,out_trans1_0));

        //step-6: finalize qnn graph and execute qnn graph
        CHECK_QNN_API(error, qnn_raw_interface.graphFinalize(graph_handle, nullptr, nullptr));
        Qnn_Tensor_t input_tensors_0[]  = {*p_tensor0, *p_tensor1};
        Qnn_Tensor_t output_tensors_0[] = {*p_tensor2};
        CHECK_QNN_API(error, qnn_raw_interface.graphExecute(graph_handle,
                                                            input_tensors_0, 2,
                                                            output_tensors_0, 1,
                                                            nullptr, nullptr));

        qnn_tensors_t ggml_op_mulmat_tensors;
        ggml_op_mulmat_tensors.reserve(5);
        ggml_op_mulmat_tensors.push_back(p_tensor0);
        ggml_op_mulmat_tensors.push_back(p_tensor1);
        ggml_op_mulmat_tensors.push_back(p_tensor2);
        ggml_op_mulmat_tensors.push_back(p_param_tensor);
        ggml_op_mulmat_tensors.push_back(p_tensor2_transpose);
        auto  graph_item = std::make_tuple(graph_handle, ggml_op_mulmat_tensors);
        instance->_qnn_graph_map[graph_name] = graph_item;
    } else {
        if (src0_type != GGML_TYPE_F32) {
            QNN_VER_PTR(*p_tensor0)->clientBuf = {wdata, static_cast<uint32_t>(desired_size)};
        } else {
            QNN_VER_PTR(*p_tensor0)->clientBuf = {src0->data, static_cast<uint32_t>(ggml_nbytes(src0))};
        }
        QNN_VER_PTR(*p_tensor1)->clientBuf = {src1->data, static_cast<uint32_t>(ggml_nbytes(src1))};
        QNN_VER_PTR(*p_tensor2)->clientBuf = {dst->data, static_cast<uint32_t>(ggml_nbytes(dst))};

        Qnn_Tensor_t tensor_inputs[] = {
                *p_tensor0,
                *p_tensor1
        };
        Qnn_Tensor_t tensor_outputs[] = {
                *p_tensor2
        };
        // this is the second technical approach or another pipeline of "how to utilize the Hexagon
        // NPU maximally" through QNN SDK, details could be found at
        // https://github.com/ggml-org/llama.cpp/pull/12049#issuecomment-2678308360
        CHECK_QNN_API(error, qnn_raw_interface.graphExecute(graph_handle,
                                                            tensor_inputs, 2,
                                                            tensor_outputs, 1,
                                                            nullptr, nullptr));
    }

    // restore the original dimensions of qnn tensors to avoid memory leak in func free_qnn_tensor
    QNN_VER_PTR(*p_tensor0)->dimensions = tensor_0_dimensions;
    QNN_VER_PTR(*p_tensor1)->dimensions = tensor_1_dimensions;
    QNN_VER_PTR(*p_tensor2)->dimensions = tensor_2_dimensions;
    op_perf.info();
}
void ggml_qnn_repeat(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_add(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_div(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_leaky_relu(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_concat(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_arange(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_sqr(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_clamp(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_scale(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_argsort(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_norm(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_group_norm(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_acc(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_sum_rows(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_upsample_nearest2d(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_pad(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

static void ggml_qnn_avg_pool2d(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

static void ggml_qnn_max_pool2d(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_pool2d(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_dup(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_rms_norm(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_diag_mask(ggml_backend_qnn_context * ctx, ggml_tensor * dst, float value) {
}

void ggml_qnn_im2col(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_timestep_embedding(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_cpy(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
    ggml_qnn_dup(ctx, dst);
}

void ggml_qnn_softmax(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_get_rows(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

void ggml_qnn_rope(ggml_backend_qnn_context * ctx, ggml_tensor * dst) {
}

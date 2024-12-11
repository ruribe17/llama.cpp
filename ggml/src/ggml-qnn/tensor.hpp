
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "ggml-qnn.h"

#include "buffer.hpp"
#include "logger.hpp"
#include "qnn-lib.hpp"
#include "utils.hpp"

namespace qnn {

static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS should be 4");

class ggml_qnn_tensor {
public:
    typedef enum _tensor_type { INPUT, OUTPUT, INTERMEDIATE, PARAMETER } tensor_type_t;

    explicit ggml_qnn_tensor(tensor_type_t tensor_type, const std::string &name,
                             const qnn_dimension_array_t &dimensions, Qnn_DataType_t data_type, int rank,
                             QNNBackend device, Qnn_GraphHandle_t graph_handle,
                             std::shared_ptr<qnn_instance> qnn_instance)
        : _tensor_name(name), _device(device), _qnn_instance(qnn_instance), _graph_handle(graph_handle) {
        if (!_tensor_name.empty()) {
            QNN_TENSOR_SET_NAME(_qnn_tensor, _tensor_name.c_str());
        }

        _dimensions = dimensions;
        QNN_TENSOR_SET_DIMENSIONS(_qnn_tensor, _dimensions.data());
        QNN_TENSOR_SET_DATA_FORMAT(_qnn_tensor, QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER);
        update_params_from_ggml_tensor(tensor_type, data_type, rank);
        QNN_LOG_DEBUG("[%s][%s]created, rank: %d, dims: [%d, %d, %d, %d], type: %s", get_backend_name(device),
                      _tensor_name.c_str(), rank, (int)_dimensions[0], (int)_dimensions[1], (int)_dimensions[2],
                      (int)_dimensions[3], qnn_datatype_to_string(data_type));
    }

    explicit ggml_qnn_tensor(tensor_type_t tensor_type, const std::string &name,
                             const ggml_dimension_array_t &dimensions, ggml_type data_type, int rank, QNNBackend device,
                             Qnn_GraphHandle_t graph_handle, std::shared_ptr<qnn_instance> qnn_instance)
        : ggml_qnn_tensor(tensor_type, name, get_internal_dimension(dimensions, rank),
                          qnn_datatype_from_ggml_datatype(data_type), rank, device, graph_handle, qnn_instance) {}

    ~ggml_qnn_tensor() {
        _buffer_storage.clear();
        unbind();
        _rpc_buffer.reset();
    }

    bool set_data_buffer(std::vector<uint8_t> &&buffer) {
        if (!bind_buffer_impl(buffer.data(), buffer.size())) {
            return false;
        }

        _buffer_storage = std::move(buffer);
        return true;
    }

    bool alloc_qnn_tensor_id() {
        if (QNN_TENSOR_GET_ID(_qnn_tensor)) {
            QNN_LOG_DEBUG("[%s]tensor already has a id: %d", _tensor_name.c_str(), QNN_TENSOR_GET_ID(_qnn_tensor));
            return true;
        }

        Qnn_Tensor_t qnn_tensor = _qnn_tensor;
        auto qnn_interface = _qnn_instance->get_qnn_interface();
        auto error = qnn_interface->qnn_tensor_create_graph_tensor(_graph_handle, &qnn_tensor);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("[%s]allocate id failed , error: %d", _tensor_name.c_str(), error);
            return false;
        }

        QNN_TENSOR_SET_ID(_qnn_tensor, QNN_TENSOR_GET_ID(qnn_tensor));
        QNN_LOG_DEBUG("[%s][%s]allocated id: %d, rank: %d", get_backend_name(_device), _tensor_name.c_str(),
                      QNN_TENSOR_GET_ID(qnn_tensor), QNN_TENSOR_GET_RANK(qnn_tensor));
        return true;
    }

    bool bind_buffer(uint8_t *buffer, const size_t buffer_size) {
        if (!_buffer_storage.empty()) {
            QNN_LOG_DEBUG("[%s]already has buffer storage, skip bind", _tensor_name.c_str());
            return true;
        }

        return bind_buffer_impl(buffer, buffer_size);
    }

    bool bind_ggml_tensor(ggml_tensor *tensor) {
        if (!bind_buffer(reinterpret_cast<uint8_t *>(tensor->data), ggml_nbytes(tensor))) {
            QNN_LOG_WARN("[%s]failed to bind ggml tensor(%s)", _tensor_name.c_str(), ggml_get_name(tensor));
            return false;
        }

        QNN_LOG_DEBUG("[%s][%s]bind to ggml tensor(%s)", get_backend_name(_device), _tensor_name.c_str(),
                      ggml_get_name(tensor));
        return true;
    }

    bool unbind() {
        if (!_graph_handle) {
            QNN_LOG_WARN("[%s]not bound to any graph", _tensor_name.c_str());
            return false;
        }

        if (!_buffer) {
            QNN_LOG_DEBUG("[%s]bound to ggml tensor", _tensor_name.c_str());
            return true;
        }

        if (!read_from_qnn_tensor()) {
            QNN_LOG_WARN("[%s]read from qnn tensor failed", _tensor_name.c_str());
            return false;
        }

        if (!_buffer_storage.empty()) {
            QNN_LOG_DEBUG("[%s]already has buffer storage, stop unbind", _tensor_name.c_str());
            return true;
        }

        if (!should_use_mem_handle()) {
            QNN_TENSOR_SET_MEM_TYPE(_qnn_tensor, QNN_TENSORMEMTYPE_RAW);
            Qnn_ClientBuffer_t client_buf = {};
            QNN_TENSOR_SET_CLIENT_BUF(_qnn_tensor, client_buf);
            QNN_LOG_DEBUG("[%s]clear client buffer", _tensor_name.c_str());
        }

        QNN_LOG_DEBUG("[%s][%s]unbind from buffer: %p, size: %d", get_backend_name(_device), _tensor_name.c_str(),
                      _buffer, (int)_buffer_size);
        _buffer = nullptr;
        _buffer_size = 0;
        return true;
    }

    const Qnn_Tensor_t &get_qnn_tensor() const { return _qnn_tensor; }
    Qnn_DataType_t get_data_type() const { return QNN_TENSOR_GET_DATA_TYPE(_qnn_tensor); }
    const qnn_dimension_array_t &get_dimensions() const { return _dimensions; }
    uint32_t get_qnn_tensor_id() const { return QNN_TENSOR_GET_ID(_qnn_tensor); }

private:
    bool bind_buffer_impl(uint8_t *buffer, const size_t buffer_size) {
        if (_buffer) {
            if (_buffer != buffer) {
                QNN_LOG_WARN("[%s]has been bound to another buffer %p", _tensor_name.c_str(), _buffer);
                return false;
            }

            QNN_LOG_DEBUG("[%s]already bound to same ggml tensor %p", _tensor_name.c_str(), _buffer);
            return true;
        }

        if (QNN_TENSOR_GET_TYPE(_qnn_tensor) == QNN_TENSOR_TYPE_NATIVE) {
            QNN_LOG_DEBUG("[%s]tensor type(%d) not READ/WRITE, skipping", _tensor_name.c_str(),
                          (int)QNN_TENSOR_TYPE_NATIVE);
            return true;
        }

        if (should_use_mem_handle()) {
            if (!_rpc_buffer) {
                auto rpc_buffer = std::make_shared<qnn_rpc_buffer>(
                    _qnn_instance, buffer_size, QNN_TENSOR_GET_RANK(_qnn_tensor),
                    QNN_TENSOR_GET_DIMENSIONS(_qnn_tensor), QNN_TENSOR_GET_DATA_TYPE(_qnn_tensor));
                if (!rpc_buffer->is_valid()) {
                    QNN_LOG_WARN("[%s][%s]alloc rpc mem failed", get_backend_name(_device), _tensor_name.c_str());
                    return false;
                }

                _rpc_buffer = std::move(rpc_buffer);
            }

            QNN_TENSOR_SET_MEM_TYPE(_qnn_tensor, QNN_TENSORMEMTYPE_MEMHANDLE);
            auto mem_handle = _rpc_buffer->get_mem_handle();
            if (!mem_handle) {
                QNN_LOG_WARN("[%s][%s]can't find rpcmem from qnn mem handle", get_backend_name(_device),
                             _tensor_name.c_str());
                return false;
            }

            QNN_TENSOR_SET_MEM_HANDLE(_qnn_tensor, mem_handle);
            QNN_LOG_DEBUG("[%s][%s]use mem handle %p", get_backend_name(_device), _tensor_name.c_str(),
                          QNN_TENSOR_GET_MEM_HANDLE(_qnn_tensor));
        } else {
            QNN_TENSOR_SET_MEM_TYPE(_qnn_tensor, QNN_TENSORMEMTYPE_RAW);
            Qnn_ClientBuffer_t client_buf = {buffer, (uint32_t)buffer_size};
            QNN_TENSOR_SET_CLIENT_BUF(_qnn_tensor, client_buf);
            QNN_LOG_DEBUG("[%s]use client buffer %p size %d", _tensor_name.c_str(), client_buf.data,
                          (int)client_buf.dataSize);
        }

        _buffer = buffer;
        _buffer_size = buffer_size;

        if (!write_to_qnn_tensor()) {
            QNN_LOG_WARN("[%s]write to qnn tensor failed", _tensor_name.c_str());
            return false;
        }

        QNN_LOG_DEBUG("[%s][%s]bind to buffer: %p, size: %d", get_backend_name(_device), _tensor_name.c_str(), buffer,
                      (int)buffer_size);
        return true;
    }

    bool write_to_qnn_tensor() {
        auto tensor_type = QNN_TENSOR_GET_TYPE(_qnn_tensor);
        if (tensor_type != QNN_TENSOR_TYPE_APP_WRITE && tensor_type != QNN_TENSOR_TYPE_APP_READWRITE) {
            QNN_LOG_DEBUG("[%s]tensor type(%d) not WRITE", _tensor_name.c_str(), (int)tensor_type);
            return true;
        }

        if (_rpc_buffer) {
            memcpy(_rpc_buffer->get_buffer(), _buffer, _buffer_size);
        }

        // For CPU and GPU, the data is already in the tensor.
        QNN_LOG_DEBUG("[%s][%s]write tensor to qnn", get_backend_name(_device), _tensor_name.c_str());
        return true;
    }

    bool read_from_qnn_tensor() {
        auto tensor_type = QNN_TENSOR_GET_TYPE(_qnn_tensor);
        if (tensor_type != QNN_TENSOR_TYPE_APP_READ && tensor_type != QNN_TENSOR_TYPE_APP_READWRITE) {
            QNN_LOG_DEBUG("[%s]tensor type(%d) not READ", _tensor_name.c_str(), (int)tensor_type);
            return true;
        }

        if (_rpc_buffer) {
            memcpy(_buffer, _rpc_buffer->get_buffer(), _buffer_size);
        }

        // For CPU and GPU, the data is already in the tensor.
        QNN_LOG_DEBUG("[%s][%s]read tensor from qnn", get_backend_name(_device), _tensor_name.c_str());
        return true;
    }

    void update_params_from_ggml_tensor(tensor_type_t tensor_type, Qnn_DataType_t data_type, int rank) {
        QNN_TENSOR_SET_DATA_TYPE(_qnn_tensor, data_type);
        // TODO: set the quantizeParams base on the tensor type

        QNN_TENSOR_SET_RANK(_qnn_tensor, (uint32_t)rank);
        QNN_TENSOR_SET_MEM_TYPE(_qnn_tensor, QNN_TENSORMEMTYPE_RAW);
        Qnn_ClientBuffer_t client_buf = {};
        QNN_TENSOR_SET_CLIENT_BUF(_qnn_tensor, client_buf);

        Qnn_TensorType_t new_tensor_type;
        switch (tensor_type) {
            case INPUT:
                new_tensor_type = QNN_TENSOR_TYPE_APP_WRITE;
                break;
            case OUTPUT:
                new_tensor_type = QNN_TENSOR_TYPE_APP_READ;
                break;
            case PARAMETER:
                new_tensor_type = QNN_TENSOR_TYPE_STATIC;
                break;
            case INTERMEDIATE:
            default:
                new_tensor_type = QNN_TENSOR_TYPE_NATIVE;
                break;
        }
        QNN_TENSOR_SET_TYPE(_qnn_tensor, new_tensor_type);
        QNN_LOG_DEBUG("[%s][%s]tensor changed to type %d", get_backend_name(_device), _tensor_name.c_str(),
                      new_tensor_type);
    }

    bool should_use_mem_handle() const {
        return _device == QNN_BACKEND_NPU && QNN_TENSOR_GET_TYPE(_qnn_tensor) != QNN_TENSOR_TYPE_STATIC;
    }

    std::string _tensor_name;
    uint8_t *_buffer = nullptr;
    size_t _buffer_size = 0;
    std::vector<uint8_t> _buffer_storage;
    QNNBackend _device;
    std::shared_ptr<qnn_instance> _qnn_instance;
    Qnn_Tensor_t _qnn_tensor = qnn_tensor_init(kDefaultQnnTensorVersion);
    qnn_dimension_array_t _dimensions = {};
    Qnn_GraphHandle_t _graph_handle = nullptr;
    qnn_buffer_ptr _rpc_buffer;

    DISABLE_COPY(ggml_qnn_tensor);
    DISABLE_MOVE(ggml_qnn_tensor);
};

using qnn_tensor_ptr_t = std::shared_ptr<ggml_qnn_tensor>;
using qnn_tensor_array_t = std::vector<qnn_tensor_ptr_t>;

} // namespace qnn

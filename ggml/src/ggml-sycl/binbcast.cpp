#include "binbcast.hpp"
#include "common.hpp"

static __dpct_inline__ float op_repeat(const float a, const float b) {
    return b;
    GGML_UNUSED(a);
}

static __dpct_inline__ float op_add(const float a, const float b) {
    return a + b;
}

static __dpct_inline__ float op_sub(const float a, const float b) {
    return a - b;
}

static __dpct_inline__ float op_mul(const float a, const float b) {
    return a * b;
}

static __dpct_inline__ float op_div(const float a, const float b) {
    return a / b;
}

template <float (*bin_op)(const float, const float), typename src0_t, typename src1_t, typename dst_t>
static void k_bin_bcast(const src0_t * src0, const src1_t * src1, dst_t * dst, int ne0, int ne1, int ne2, int ne3,
                        int ne10, int ne11, int ne12, int ne13,
                        /*int s0, */ int s1, int s2, int s3,
                        /*int s10,*/ int s11, int s12, int s13, const sycl::nd_item<3> & item_ct1) {
    const int i0s = item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);
    const int i1  = (item_ct1.get_local_range(1) * item_ct1.get_group(1) + item_ct1.get_local_id(1));
    const int i2  = (item_ct1.get_local_range(0) * item_ct1.get_group(0) + item_ct1.get_local_id(0)) / ne3;
    const int i3  = (item_ct1.get_local_range(0) * item_ct1.get_group(0) + item_ct1.get_local_id(0)) % ne3;

    if (i0s >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        return;
    }

    const int i11 = i1 % ne11;
    const int i12 = i2 % ne12;
    const int i13 = i3 % ne13;

    const size_t i_src0 = i3 * s3 + i2 * s2 + i1 * s1;
    const size_t i_src1 = i13 * s13 + i12 * s12 + i11 * s11;
    const size_t i_dst  = i_src0;

    const src0_t * src0_row = src0 + i_src0;
    const src1_t * src1_row = src1 + i_src1;
    dst_t *        dst_row  = dst + i_dst;

    for (int i0 = i0s; i0 < ne0; i0 += item_ct1.get_local_range(2) * item_ct1.get_group_range(2)) {
        const int i10 = i0 % ne10;
        dst_row[i0]   = (dst_t) bin_op(src0 ? (float) src0_row[i0] : 0.0f, (float) src1_row[i10]);
    }
}

template <float (*bin_op)(const float, const float), typename src0_t, typename src1_t, typename dst_t>
static void k_bin_bcast_unravel(const src0_t * src0, const src1_t * src1, dst_t * dst, int ne0, int ne1, int ne2,
                                int ne3, int ne10, int ne11, int ne12, int ne13,
                                /*int s0, */ int s1, int s2, int s3,
                                /*int s10,*/ int s11, int s12, int s13, const sycl::nd_item<3> & item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);

    const int i3 = i / (ne2 * ne1 * ne0);
    const int i2 = (i / (ne1 * ne0)) % ne2;
    const int i1 = (i / ne0) % ne1;
    const int i0 = i % ne0;

    if (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        return;
    }

    const int i11 = i1 % ne11;
    const int i12 = i2 % ne12;
    const int i13 = i3 % ne13;

    const size_t i_src0 = i3 * s3 + i2 * s2 + i1 * s1;
    const size_t i_src1 = i13 * s13 + i12 * s12 + i11 * s11;
    const size_t i_dst  = i_src0;

    const src0_t * src0_row = src0 + i_src0;
    const src1_t * src1_row = src1 + i_src1;
    dst_t *        dst_row  = dst + i_dst;

    const int i10 = i0 % ne10;
    dst_row[i0]   = (dst_t) bin_op(src0 ? (float) src0_row[i0] : 0.0f, (float) src1_row[i10]);
}

template <float (*bin_op)(const float, const float)> struct bin_bcast_sycl {
    template <typename src0_t, typename src1_t, typename dst_t>
    void operator()(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst,
                    const src0_t * src0_dd, const src1_t * src1_dd, dst_t * dst_dd, queue_ptr stream) {
        GGML_TENSOR_BINARY_OP_LOCALS

        int nr0 = ne10 / ne0;
        int nr1 = ne11 / ne1;
        int nr2 = ne12 / ne2;
        int nr3 = ne13 / ne3;

        int nr[4] = { nr0, nr1, nr2, nr3 };

        // collapse dimensions until first broadcast dimension
        int64_t cne0[]   = { ne0, ne1, ne2, ne3 };
        int64_t cne1[]   = { ne10, ne11, ne12, ne13 };
        size_t  cnb0[]   = { nb0, nb1, nb2, nb3 };
        size_t  cnb1[]   = { nb10, nb11, nb12, nb13 };
        auto    collapse = [](int64_t cne[]) {
            cne[0] *= cne[1];
            cne[1] = cne[2];
            cne[2] = cne[3];
            cne[3] = 1;
        };

        auto collapse_nb = [](size_t cnb[], int64_t cne[]) {
            cnb[1] *= cne[1];
            cnb[2] *= cne[2];
            cnb[3] *= cne[3];
        };

        for (int i = 0; i < 4; i++) {
            if (nr[i] != 1) {
                break;
            }
            if (i > 0) {
                collapse_nb(cnb0, cne0);
                collapse_nb(cnb1, cne1);
                collapse(cne0);
                collapse(cne1);
            }
        }
        {
            int64_t ne0 = cne0[0];
            int64_t ne1 = cne0[1];
            int64_t ne2 = cne0[2];
            int64_t ne3 = cne0[3];

            int64_t ne10 = cne1[0];
            int64_t ne11 = cne1[1];
            int64_t ne12 = cne1[2];
            int64_t ne13 = cne1[3];

            size_t nb0 = cnb0[0];
            size_t nb1 = cnb0[1];
            size_t nb2 = cnb0[2];
            size_t nb3 = cnb0[3];

            size_t nb10 = cnb1[0];
            size_t nb11 = cnb1[1];
            size_t nb12 = cnb1[2];
            size_t nb13 = cnb1[3];

            size_t s0 = nb0 / sizeof(dst_t);
            size_t s1 = nb1 / sizeof(dst_t);
            size_t s2 = nb2 / sizeof(dst_t);
            size_t s3 = nb3 / sizeof(dst_t);

            size_t s10 = nb10 / sizeof(src1_t);
            size_t s11 = nb11 / sizeof(src1_t);
            size_t s12 = nb12 / sizeof(src1_t);
            size_t s13 = nb13 / sizeof(src1_t);

            GGML_ASSERT(s0 == 1);
            GGML_ASSERT(s10 == 1);

            const int block_size = 128;

            int64_t hne0 = std::max(ne0 / 2LL, 1LL);

            sycl::range<3> block_dims(1, 1, 1);
            block_dims[2] = std::min<unsigned int>(hne0, block_size);
            block_dims[1] = std::min<unsigned int>(ne1, block_size / (unsigned int) block_dims[2]);
            block_dims[0] = std::min(std::min<unsigned int>(ne2 * ne3, block_size / (unsigned int) block_dims[2] /
                                                                           (unsigned int) block_dims[1]),
                                     64U);

            sycl::range<3> block_nums((ne2 * ne3 + block_dims[0] - 1) / block_dims[0],
                                      (ne1 + block_dims[1] - 1) / block_dims[1],
                                      (hne0 + block_dims[2] - 1) / block_dims[2]);

            if (block_nums[0] > 65535) {
                // this is the maximum number of blocks in z direction, fallback to 1D grid kernel
                int block_num = (ne0 * ne1 * ne2 * ne3 + block_size - 1) / block_size;
                {
                    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

                    stream->parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, block_num) * sycl::range<3>(1, 1, block_size),
                                          sycl::range<3>(1, 1, block_size)),
                        [=](sycl::nd_item<3> item_ct1) {
                            k_bin_bcast_unravel<bin_op>(src0_dd, src1_dd, dst_dd, ne0, ne1, ne2, ne3, ne10, ne11, ne12,
                                                        ne13, s1, s2, s3, s11, s12, s13, item_ct1);
                        });
                }
            } else {
                /*
                DPCT1049:16: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

                stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         k_bin_bcast<bin_op>(src0_dd, src1_dd, dst_dd, ne0, ne1, ne2, ne3, ne10, ne11,
                                                             ne12, ne13, s1, s2, s3, s11, s12, s13, item_ct1);
                                     });
            }
        }
    }
};

template <class op>
inline void ggml_sycl_op_bin_bcast(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
                                   const void * src0_dd, const void * src1_dd, void * dst_dd,
                                   const queue_ptr & main_stream) {
    if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        op()(src0, src1, dst, (const float *) src0_dd, (const float *) src1_dd, (float *) dst_dd, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        op()(src0, src1, dst, (const sycl::half *) src0_dd, (const float *) src1_dd, (sycl::half *) dst_dd,
             main_stream);
    } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
        op()(src0, src1, dst, (const sycl::half *) src0_dd, (const float *) src1_dd, (float *) dst_dd, main_stream);
    } else if (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I32) {
        op()(src0, src1, dst, (const int32_t *) src0_dd, (const int32_t *) src1_dd, (int32_t *) dst_dd, main_stream);
    } else if (src0->type == GGML_TYPE_I16 && dst->type == GGML_TYPE_I16) {
        op()(src0, src1, dst, (const int16_t *) src0_dd, (const int16_t *) src1_dd, (int16_t *) dst_dd, main_stream);
    } else {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__, ggml_type_name(dst->type),
                ggml_type_name(src0->type), ggml_type_name(src1->type));
        GGML_ABORT("fatal error");
    }
}

inline void ggml_sycl_op_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const void *          src0_dd     = static_cast<void *>(dst->src[0]->data);
    const void *          src1_dd     = static_cast<void *>(dst->src[1]->data);
    void *                dst_dd      = static_cast<void *>(dst->data);
    const dpct::queue_ptr main_stream = ctx.stream();

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_add>>(dst->src[0], dst->src[1], dst, src0_dd, src1_dd, dst_dd,
                                                   main_stream);
}

inline void ggml_sycl_op_sub(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const void *          src0_dd     = static_cast<void *>(dst->src[0]->data);
    const void *          src1_dd     = static_cast<void *>(dst->src[1]->data);
    void *                dst_dd      = static_cast<void *>(dst->data);
    const dpct::queue_ptr main_stream = ctx.stream();

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_sub>>(dst->src[0], dst->src[1], dst, src0_dd, src1_dd, dst_dd,
                                                   main_stream);
}

inline void ggml_sycl_op_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const void *          src0_dd     = static_cast<void *>(dst->src[0]->data);
    const void *          src1_dd     = static_cast<void *>(dst->src[1]->data);
    void *                dst_dd      = static_cast<void *>(dst->data);
    const dpct::queue_ptr main_stream = ctx.stream();

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_mul>>(dst->src[0], dst->src[1], dst, src0_dd, src1_dd, dst_dd,
                                                   main_stream);
}

inline void ggml_sycl_op_div(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const void *          src0_dd     = static_cast<void *>(dst->src[0]->data);
    const void *          src1_dd     = static_cast<void *>(dst->src[1]->data);
    void *                dst_dd      = static_cast<void *>(dst->data);
    const dpct::queue_ptr main_stream = ctx.stream();

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_div>>(dst->src[0], dst->src[1], dst, src0_dd, src1_dd, dst_dd,
                                                   main_stream);
}

inline void ggml_sycl_op_repeat(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const void *    src0_d      = static_cast<void *>(dst->src[0]->data);
    void *          dst_d       = static_cast<void *>(dst->data);
    dpct::queue_ptr main_stream = ctx.stream();

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_repeat>>(dst, dst->src[0], dst, nullptr, src0_d, dst_d, main_stream);
}

void ggml_sycl_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_add(ctx, dst);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_sub(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_sub(ctx, dst);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_mul(ctx, dst);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_div(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_div(ctx, dst);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_repeat(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_repeat(ctx, dst);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

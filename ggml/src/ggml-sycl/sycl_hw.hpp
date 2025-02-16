#ifndef SYCL_HW_HPP
#define SYCL_HW_HPP

#include <algorithm>
#include <stdio.h>
#include <vector>
#include <map>

#include <sycl/sycl.hpp>

enum SYCL_HW_FAMILY {
  SYCL_HW_FAMILY_UNKNOWN = -1,
  SYCL_HW_FAMILY_INTEL_UHD = 0,
  SYCL_HW_FAMILY_INTEL_IRIS = 10,
  SYCL_HW_FAMILY_INTEL_ARC = 20,
  SYCL_HW_FAMILY_INTEL_MTL_ARL = 30,
  SYCL_HW_FAMILY_INTEL_LNL = 31,
  SYCL_HW_FAMILY_INTEL_PVC = 60,
};

struct sycl_hw_info {
  int family;
  int32_t device_id;
};

bool is_in_vector(std::vector<int> &vec, int item);

sycl_hw_info get_device_hw_info(sycl::device *device_ptr);

#endif // SYCL_HW_HPP

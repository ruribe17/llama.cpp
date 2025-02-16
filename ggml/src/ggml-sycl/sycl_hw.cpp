#include "sycl_hw.hpp"

bool is_in_vector(const std::vector<int> &vec, int item) {
  return std::find(vec.begin(), vec.end(), item) != vec.end();
}

void search_device_info(std::map<std::vector<int>, std::vector<int>> &device_map,
  int32_t id_prefix, sycl_hw_info *res){
  for (const auto& [ids, infos] : device_map) {
    std::vector<int> tmp_ids;
    for (const auto& device_id : ids) {
      tmp_ids.push_back(device_id & 0xff00);
    }

    if (is_in_vector((const std::vector<int>)tmp_ids, id_prefix)) {
      res->family = infos[0];
      return;
    }
  }
  res->family = SYCL_HW_FAMILY_UNKNOWN;
}

sycl_hw_info get_device_hw_info(sycl::device *device_ptr) {
  sycl_hw_info res;
  int32_t id = device_ptr->get_info<sycl::ext::intel::info::device::device_id>();
  res.device_id = id;
  int32_t id_prefix = id & 0xff00;

  std::map<std::vector<int>, std::vector<int>> device_map = {
    {{0x4600}, {SYCL_HW_FAMILY_INTEL_UHD}},
    {{0x4900, 0xa700}, {SYCL_HW_FAMILY_INTEL_IRIS}},
    {{0x5600, 0x4f00}, {SYCL_HW_FAMILY_INTEL_ARC}},
    {{0x7D00}, {SYCL_HW_FAMILY_INTEL_MTL_ARL}},
    {{0x6400}, {SYCL_HW_FAMILY_INTEL_LNL}},
    {{0x0B00}, {SYCL_HW_FAMILY_INTEL_PVC}},
  };

  search_device_info(device_map, id_prefix, &res);
  return res;
}

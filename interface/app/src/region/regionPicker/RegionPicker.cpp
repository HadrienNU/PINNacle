#include <region/regionPicker/RegionPicker.hpp>


RegionPicker::RegionPicker() {}

void RegionPicker::build(const std::vector<int>& regionIds, const std::vector<std::vector<glm::vec3>>& regionMeshes) {
    _bvh.build(regionIds, regionMeshes);
}

int RegionPicker::pick(const glm::vec3& rayOrigin, const glm::vec3& rayDir) const {
    RayHit hit = _bvh.intersect(rayOrigin, rayDir);
    return hit.hit ? hit.regionId : -1;
}

#ifndef __REGION_PICKER_HPP__
#define __REGION_PICKER_HPP__


#include <region/regionPicker/BVH.hpp>
#include <util.hpp>
#include <glm/glm.hpp>
#include <vector>


class RegionPicker {
public:
    RegionPicker();
    ~RegionPicker() = default;
    void build(const std::vector<int>& regionIds, const std::vector<std::vector<glm::vec3>>& regionMeshes);
    int pick(const glm::vec3& rayOrigin, const glm::vec3& rayDir) const;
private:
    BVH _bvh;
};

#endif

#ifndef __BVH_HPP__
#define __BVH_HPP__


#include <region/regionPicker/AABB.hpp>
#include <region/regionPicker/Triangle.hpp>
#include <region/regionPicker/RayHit.hpp>
#include <region/regionPicker/BVHNode.hpp>
#include <glm/glm.hpp>
#include <vector>

#define VERTICES_PER_TRIANGLE 3
#define MAX_LEAF_TRIANGLES 4


class BVH {
public:
    BVH();
    ~BVH() = default;
    void build(const std::vector<int>& regionIds, const std::vector<std::vector<glm::vec3>>& regionMeshes);
    RayHit intersect(const glm::vec3& rayOrigin, const glm::vec3& rayDir) const;
private:
    int buildRecursive(int start, int end);
    RayHit intersectRecursive(int nodeIndex, const glm::vec3& rayOrigin, const glm::vec3& rayDir) const;
private:
    std::vector<BVHNode> _nodes;
    std::vector<Triangle> _triangles;
};

#endif

#include <region/regionPicker/BVH.hpp>
#include <algorithm>


BVH::BVH() {}

void BVH::build(const std::vector<int>& regionIds, const std::vector<std::vector<glm::vec3>>& regionMeshes) {
    _triangles.clear();
    _nodes.clear();
    
    if (regionIds.empty() || regionMeshes.empty()) {
        return;
    }
    
    for (size_t i = 0; i < regionIds.size(); i++) {
        const std::vector<glm::vec3>& mesh = regionMeshes[i];
        int regionId = regionIds[i];
        
        for (size_t j = 0; j + VERTICES_PER_TRIANGLE - 1 < mesh.size(); j += VERTICES_PER_TRIANGLE) {
            _triangles.emplace_back(mesh[j], mesh[j + 1], mesh[j + 2], regionId);
        }
    }
    
    if (_triangles.empty()) {
        return;
    }
    
    _nodes.reserve(_triangles.size() * 2);
    buildRecursive(0, _triangles.size());
}

int BVH::buildRecursive(int start, int end) {
    int nodeIndex = _nodes.size();
    _nodes.emplace_back();
    BVHNode& node = _nodes[nodeIndex];
    
    AABB bounds;
    for (int i = start; i < end; i++) {
        bounds.expand(_triangles[i].getBounds());
    }
    node.bounds = bounds;
    
    int count = end - start;
    if (count <= MAX_LEAF_TRIANGLES) {
        node.firstTriangle = start;
        node.triangleCount = count;
        return nodeIndex;
    }
    
    glm::vec3 extent = bounds.max - bounds.min;
    int axis = 0;
    if (extent.y > extent.x) {
        axis = 1;
    }
    if (extent.z > extent[axis]) {
        axis = 2;
    }
    float splitPos = bounds.min[axis] + extent[axis] * 0.5f;
    
    int mid = start;
    for (int i = start; i < end; i++) {
        glm::vec3 centroid = _triangles[i].getBounds().center();
        if (centroid[axis] < splitPos) {
            std::swap(_triangles[i], _triangles[mid]);
            mid++;
        }
    }
    
    if (mid == start || mid == end) {
        mid = start + count / 2;
    }
    
    node.leftChild = buildRecursive(start, mid);
    node.rightChild = buildRecursive(mid, end);
    
    return nodeIndex;
}

RayHit BVH::intersect(const glm::vec3& rayOrigin, const glm::vec3& rayDir) const {
    if (_nodes.empty() || _triangles.empty()) {
        return RayHit();
    }
    
    return intersectRecursive(0, rayOrigin, rayDir);
}

RayHit BVH::intersectRecursive(int nodeIndex, const glm::vec3& rayOrigin, const glm::vec3& rayDir) const {
    RayHit result;
    
    if (nodeIndex < 0 || nodeIndex >= static_cast<int>(_nodes.size())) {
        return result;
    }
    
    const BVHNode& node = _nodes[nodeIndex];
    
    float tMin, tMax;
    if (!node.bounds.intersect(rayOrigin, rayDir, tMin, tMax)) {
        return result;
    }
    
    if (node.isLeaf()) {
        for (int i = 0; i < node.triangleCount; i++) {
            const Triangle& tri = _triangles[node.firstTriangle + i];
            float t;
            if (tri.intersect(rayOrigin, rayDir, t)) {
                if (t < result.t) {
                    result.hit = true;
                    result.t = t;
                    result.regionId = tri.regionId;
                }
            }
        }
        return result;
    }
    
    RayHit hitLeft = intersectRecursive(node.leftChild, rayOrigin, rayDir);
    RayHit hitRight = intersectRecursive(node.rightChild, rayOrigin, rayDir);
    
    if (hitLeft.hit && hitRight.hit) {
        return hitLeft.isCloser(hitRight) ? hitLeft : hitRight;
    } else if (hitLeft.hit) {
        return hitLeft;
    } else if (hitRight.hit) {
        return hitRight;
    }
    
    return result;
}

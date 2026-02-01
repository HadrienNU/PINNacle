#ifndef __TRIANGLE_HPP__
#define __TRIANGLE_HPP__


#include <glm/glm.hpp>
#include <region/regionPicker/AABB.hpp>

#define TRIANGLE_EPSILON 1e-8f


struct Triangle {
    glm::vec3 v0, v1, v2;
    int regionId;
    Triangle(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2, int regionId);
    AABB getBounds() const;
    bool intersect(const glm::vec3& rayOrigin, const glm::vec3& rayDir, float& t) const;
};

#endif

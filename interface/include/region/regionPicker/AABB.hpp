#ifndef __AABB_HPP__
#define __AABB_HPP__


#include <glm/glm.hpp>
#include <limits>

#define DIMENSION 3
#define EPSILON 1e-8f


struct AABB {
    glm::vec3 min;
    glm::vec3 max;
    AABB();
    AABB(const glm::vec3& min, const glm::vec3& max);
    void expand(const glm::vec3& point);
    void expand(const AABB& other);
    glm::vec3 center() const;
    bool intersect(const glm::vec3& rayOrigin, const glm::vec3& rayDir, float& tMin, float& tMax) const;
};

#endif

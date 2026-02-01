#include <region/regionPicker/AABB.hpp>
#include <algorithm>
#include <cmath>


AABB::AABB() 
    : min(std::numeric_limits<float>::max()), 
      max(std::numeric_limits<float>::lowest()) {}

AABB::AABB(const glm::vec3& min, const glm::vec3& max) 
    : min(min), max(max) {}

void AABB::expand(const glm::vec3& point) {
    min = glm::min(min, point);
    max = glm::max(max, point);
}

void AABB::expand(const AABB& other) {
    min = glm::min(min, other.min);
    max = glm::max(max, other.max);
}

glm::vec3 AABB::center() const {
    return (min + max) * 0.5f;
}

bool AABB::intersect(const glm::vec3& rayOrigin, const glm::vec3& rayDir, float& tMin, float& tMax) const {
    tMin = 0.0f;
    tMax = std::numeric_limits<float>::max();

    for (int i = 0; i < DIMENSION; i++) {
        if (std::abs(rayDir[i]) < EPSILON) {
            if (rayOrigin[i] < min[i] || rayOrigin[i] > max[i]) {
                return false;
            }
        } else {
            float t1 = (min[i] - rayOrigin[i]) / rayDir[i];
            float t2 = (max[i] - rayOrigin[i]) / rayDir[i];
            
            if (t1 > t2) {
                std::swap(t1, t2);
            }
            
            tMin = std::max(tMin, t1);
            tMax = std::min(tMax, t2);
            
            if (tMin > tMax) {
                return false;
            }
        }
    }
    
    return tMax >= 0.0f;
}

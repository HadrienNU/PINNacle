#include <region/regionPicker/Triangle.hpp>


Triangle::Triangle(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2, int regionId)
    : v0(v0), v1(v1), v2(v2), regionId(regionId) {}

AABB Triangle::getBounds() const {
    AABB bounds;
    bounds.expand(v0);
    bounds.expand(v1);
    bounds.expand(v2);
    return bounds;
}

bool Triangle::intersect(const glm::vec3& rayOrigin, const glm::vec3& rayDir, float& t) const {
    glm::vec3 edge1 = v1 - v0;
    glm::vec3 edge2 = v2 - v0;
    
    glm::vec3 h = glm::cross(rayDir, edge2);
    float a = glm::dot(edge1, h);

    if (a > -TRIANGLE_EPSILON && a < TRIANGLE_EPSILON) {
        return false;
    }
    
    float f = 1.0f / a;
    glm::vec3 s = rayOrigin - v0;
    float u = f * glm::dot(s, h);

    if (u < 0.0f || u > 1.0f) {
        return false;
    }
    
    glm::vec3 q = glm::cross(s, edge1);
    float v = f * glm::dot(rayDir, q);
    
    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }
    
    t = f * glm::dot(edge2, q);
    
    return t > TRIANGLE_EPSILON;
}

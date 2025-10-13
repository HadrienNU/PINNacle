#include <region/Region.hpp>

Region::Region() {}

void Region::addPoint(const glm::vec2 & point) {
    addPoint(glm::vec3(point, 1.0f));
}

void Region::addPoint(const glm::vec3 & point) {
    _points.push_back(point);
}

glm::vec3 Region::getCenter() const {
    glm::vec3 center = glm::vec3(0.0f, 0.0f, 0.0f);
    for (const glm::vec3 & point: _points) {
        center += point;
    }
    center /= _points.size();
    return center;
}

std::vector<glm::vec3> Region::createMesh() {
    std::vector<glm::vec3> vertices;
    glm::vec3 centerRegion = getCenter();
    float epsilon = 1e-2;
    
    vertices.push_back(centerRegion + glm::vec3(-epsilon, epsilon, 0.0f));
    vertices.push_back(centerRegion + glm::vec3(epsilon, epsilon, 0.0f));
    vertices.push_back(centerRegion + glm::vec3(epsilon, -epsilon, 0.0f));
    
    vertices.push_back(centerRegion + glm::vec3(-epsilon, epsilon, 0.0f));    
    vertices.push_back(centerRegion + glm::vec3(epsilon, -epsilon, 0.0f));
    vertices.push_back(centerRegion + glm::vec3(-epsilon, -epsilon, 0.0f));
    return vertices;
}


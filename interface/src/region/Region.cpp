#include <region/Region.hpp>

Region::Region() {}

void Region::addPoint(const glm::vec2 & point) {
    addPoint(glm::vec3(point, 0.0f));
}

void Region::addPoint(const glm::vec3 & point) {
    _points.push_back(point);
}

std::vector<glm::vec3> Region::createMesh() {
    std::vector<glm::vec3> vertices;
    for (size_t i = 1; i < _points.size() - 2; i ++) {
        vertices.push_back(_points[0]);
        vertices.push_back(_points[i]);
        vertices.push_back(_points[i + 1]);
    }
    return vertices;
}


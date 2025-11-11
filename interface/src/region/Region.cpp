#include <region/Region.hpp>
#include <util.hpp>

Region::Region(int id) : _id(id) {}

void Region::addPoint(const glm::vec2 & point) {
    addPoint(glm::vec3(point, 0.0f));
}

void Region::addPoint(const glm::vec3 & point) {
    _points.push_back(point);
}

std::vector<glm::vec3> Region::createMesh() {
    std::vector<glm::vec3> vertices;
    for (size_t i = 0; i < _points.size(); i +=3) {
        vertices.push_back(_points[i]);
        vertices.push_back(_points[i + 1]);
        vertices.push_back(_points[i + 2]);
    }
    return vertices;
}


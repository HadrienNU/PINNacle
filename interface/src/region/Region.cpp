#include <region/Region.hpp>
#include <util.hpp>

Region::Region(int id, int dim) : _id(id), _dim(dim) {}

void Region::addPoint(const glm::vec2 & point) {
    addPoint(glm::vec3(point, 0.0f));
}

void Region::addPoint(const glm::vec3 & point) {
    _points.push_back(point);
}

std::vector<glm::vec3> Region::createMesh() const {
    std::vector<glm::vec3> vertices;
    for (size_t i = 0; i < _points.size(); i +=3) {
        vertices.push_back(_points[i]);
        vertices.push_back(_points[i + 1]);
        vertices.push_back(_points[i + 2]);

        // 2D case, need to handle cull face
        if (_dim == 2) {
            vertices.push_back(_points[i]);
            vertices.push_back(_points[i + 2]);
            vertices.push_back(_points[i + 1]);
        }
    }
    return vertices;
}


#include <opengl/VAO.hpp>

VAO::VAO(int numVBO) {
    glGenVertexArrays(1, &_vao);
    bind();
    _vbos = new VBO[numVBO];
    unbind();
}

VAO::~VAO() {
    delete[] _vbos;
    glDeleteVertexArrays(1, &_vao);
}

void VAO::bind() {
    glBindVertexArray(_vao);
}

void VAO::unbind() {
    glBindVertexArray(0);
}

void VAO::setVector(int location, const std::vector<glm::vec3> & data) {
    bind();
    _vbos[location].bind();
    _vbos[location].setData(data.size() * sizeof(glm::vec3), data.data());
    glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(location);
    _vbos[location].unbind();
    unbind();
}


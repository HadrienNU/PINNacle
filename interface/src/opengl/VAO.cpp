#include <opengl/VAO.hpp>

VAO::VAO() {
    glGenVertexArrays(1, &_vao);
}

VAO::~VAO() {
    glDeleteVertexArrays(1, &_vao);
}

void VAO::bind() {
    glBindVertexArray(_vao);
}

void VAO::unbind() {
    glBindVertexArray(0);
}


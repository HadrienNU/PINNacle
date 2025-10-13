#include <opengl/EBO.hpp>

EBO::EBO() {
    glGenBuffers(1, &_ebo);
}

EBO::~EBO() {
    glDeleteBuffers(1, &_ebo);
}

void EBO::bind() {
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ebo);
}

void EBO::unbind() {
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}


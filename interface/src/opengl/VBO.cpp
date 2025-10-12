#include <opengl/VBO.hpp>

VBO::VBO() {
    glGenBuffers(1, &_vbo);
}

VBO::~VBO() {
    glDeleteBuffers(1, &_vbo);
}

void VBO::bind() {
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
}

void VBO::unbind() {
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}


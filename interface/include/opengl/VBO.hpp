#ifndef __VBO_HPP__
#define __VBO_HPP__

#include <util.hpp>


class VBO {
public:
    VBO();
    ~VBO();
    void bind();
    void unbind();
private:
    GLuint _vbo;
};

#endif

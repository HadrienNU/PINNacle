#ifndef __VAO_HPP__
#define __VAO_HPP__

#include <util.hpp>


class VAO {
public:
    VAO();
    ~VAO();
    void bind();
    void unbind();
private:
    GLuint _vao;
};

#endif

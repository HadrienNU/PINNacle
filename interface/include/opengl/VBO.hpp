#ifndef __VBO_HPP__
#define __VBO_HPP__

#include <util.hpp>


class VBO {
public:
    VBO();
    ~VBO();
    void bind();
    void unbind();
    void setData(size_t size, const void * data);
private:
    GLuint _vbo;
};

#endif

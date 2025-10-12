#ifndef __EBO_HPP__
#define __EBO_HPP__

#include <util.hpp>


class EBO {
public:
    EBO();
    ~EBO();
    void bind();
    void unbind();
private:
    GLuint _ebo;
};

#endif

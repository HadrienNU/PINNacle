#ifndef __VAO_HPP__
#define __VAO_HPP__

#include <opengl/VBO.hpp>
#include <util.hpp>


class VAO {
public:
    VAO(int numVBO=1);
    ~VAO();
    void bind();
    void unbind();
    void setVector(int location, const std::vector<glm::vec3> & data);
private:
    GLuint _vao;
    VBO * _vbos;
};

#endif

#ifndef __SHADER_HPP__
#define __SHADER_HPP__

#include <util.hpp>


class Shader {
public:
    Shader();
    ~Shader();
    void bind();
    void unbind();
private:
    void checkShader(GLuint shader, const String & type);
private:
    GLuint _program;
};

#endif

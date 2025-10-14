#ifndef __SHADER_HPP__
#define __SHADER_HPP__

#include <util.hpp>


class Shader {
public:
    Shader(
        const char * vertexShaderSource,
        const char * fragmentShaderSource
    );
    ~Shader();
    void bind();
    void unbind();
    void setUniformVector(const String & uniform, const glm::vec3 & vector);
private:
    void checkShader(GLuint shader, const String & type);
private:
    GLuint _program;
};

#endif

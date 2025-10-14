#ifndef __SHADER_HPP__
#define __SHADER_HPP__

#include <util.hpp>
#include <glm/gtc/type_ptr.hpp>


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
    void setUniformMatrix(const String & uniform, const glm::mat4 & matrix);
private:
    void checkShader(GLuint shader);
private:
    GLuint _program;
};

#endif

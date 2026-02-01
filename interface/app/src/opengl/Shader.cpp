#include <opengl/Shader.hpp>


Shader::Shader(
    const char * vertexShaderSource,
    const char * fragmentShaderSource
) {
    _program = glCreateProgram();

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);
    checkShader(vertexShader);

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);
    checkShader(fragmentShader);

    glAttachShader(_program, vertexShader);
    glAttachShader(_program, fragmentShader);
    glLinkProgram(_program);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

Shader::~Shader() {
    glDeleteProgram(_program);
}

void Shader::bind() {
    glUseProgram(_program);
}

void Shader::unbind() {
    glUseProgram(0);
}

void Shader::setUniformVector(const String & uniform, const glm::vec3 & vector) {
    GLint offsetLoc = glGetUniformLocation(_program, uniform.c_str());
    glUniform3f(offsetLoc, vector.x, vector.y, vector.z);
}

void Shader::setUniformMatrix(const String & uniform, const glm::mat4 & matrix) {
    GLint loc = glGetUniformLocation(_program, uniform.c_str());
    glUniformMatrix4fv(loc, 1, GL_FALSE, glm::value_ptr(matrix));
}

void Shader::setUniformFloat(const String & uniform, float value) {
    GLint loc = glGetUniformLocation(_program, uniform.c_str());
    glUniform1f(loc, value);
}

void Shader::checkShader(GLuint shader) {
    GLint success;
    GLchar infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        error(infoLog);
    }
}


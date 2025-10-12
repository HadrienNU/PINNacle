#include <opengl/Shader.hpp>


const char * vertexShaderSource = R"glsl(
#version 330 core
layout (location = 0) in vec3 aPos;

void main() {
    gl_Position = vec4(aPos, 1.0);
}
)glsl"; 

const char * fragmentShaderSource = R"glsl(
#version 330 core
out vec4 fragColor;
uniform vec3 color;

void main() {
    fragColor = vec4(color, 1);
}
)glsl";


Shader::Shader() {
    _program = glCreateProgram();

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);
    checkShader(vertexShader, "vertex");

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);
    checkShader(fragmentShader, "fragment");

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

void Shader::checkShader(GLuint shader, const String & type) {
    GLint success;
    GLchar infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        error(infoLog);
    }
}


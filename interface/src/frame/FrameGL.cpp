#include <frame/FrameGL.hpp>


const char * vertexShaderSource = R"glsl(
#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 cameraMatrix;

void main() {
    gl_Position = cameraMatrix * vec4(aPos, 1.0);
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


FrameGL::FrameGL(const Color & backgroundColor) :
_backgroundColor(backgroundColor) {
    _shader = nullptr;    
    _cameraMatrix = glm::mat4(1.0f);
}

FrameGL::~FrameGL() {
    delete _shader;
}

void FrameGL::init(const Size & frameSize) {
    _shader = new Shader(vertexShaderSource, fragmentShaderSource);
    resize(frameSize);
}

void FrameGL::resize(const Size & frameSize) {    
    _cameraMatrix = glm::perspective(
        glm::radians(DEFAULT_FOV), 
        (float) frameSize.width / frameSize.height,
         DEFAULT_NEAR_PLANE, 
         DEFAULT_FAR_PLANE
    );
    _cameraMatrix *= glm::lookAt(
        glm::vec3(0.0f, 0.0f, DEFAULT_DISTANCE), //POS
        glm::vec3(0.0f, 0.0f, 0.0f), //LOOK AT
        glm::vec3(0.0f, 1.0f, 0.0f) //UP
    );
}

void FrameGL::setRegions(const Regions & regions) {
    _regions = regions;
    size_t numberOfRegions = _regions.size();
    _tableColor = generateTableColor(numberOfRegions);
    _vaos.clear();
    for (size_t i = 0; i < numberOfRegions; i ++) {
        _vaos.push_back(std::make_unique<VAO>(1));
    }
    for (size_t i = 0; i < numberOfRegions; i ++) {
        _vaos[i] -> setVector(VERTEX_BUFFER, _regions[i].createMesh());
    }
}

void FrameGL::render() {
    glClearColor(
        _backgroundColor.r, 
        _backgroundColor.g, 
        _backgroundColor.b, 
        1.0f
    );
    glClear(GL_COLOR_BUFFER_BIT);
    _shader -> bind();
    _shader -> setUniformMatrix("cameraMatrix", _cameraMatrix);
    for (size_t i = 0; i < _regions.size(); i ++) {
        ColorGL colorGL = ColorGL(_tableColor[i]);
        glm::vec3 color(colorGL.r, colorGL.g, colorGL.b);
        _shader -> setUniformVector("color", color);
        _vaos[i] -> bind();
        glDrawArrays(GL_TRIANGLES, 0, 6);
    }   
}

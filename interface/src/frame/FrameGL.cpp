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

uniform vec3 color;

out vec4 fragColor;

void main() {
    fragColor = vec4(color, 1);
}
)glsl";


FrameGL::FrameGL(const Color & backgroundColor) :
_backgroundColor(backgroundColor) {
    _shader = nullptr;    
    _cameraMatrix = glm::mat4(1.0f);
    _distance = DEFAULT_DISTANCE;
    _yaw = 0.0f;
    _pitch = 0.0f;
   _cameraCenter = glm::vec3(0.0f, 0.0f, 0.0f);
    _aspectRatio = 1.0f;
}

FrameGL::~FrameGL() {
    delete _shader;
}

void FrameGL::init(const Size & frameSize) {
    glEnable(GL_MULTISAMPLE);
    _shader = new Shader(vertexShaderSource, fragmentShaderSource);
    resize(frameSize);
}

void FrameGL::resize(const Size & frameSize) {    
    _aspectRatio = static_cast<float>(frameSize.width) / frameSize.height;
    updateCameraMatrix();
}

void FrameGL::updateCameraMatrix() {
    const float pitchLimit = glm::radians(89.0f);
    _pitch = glm::clamp(_pitch, -pitchLimit, pitchLimit);

    glm::vec3 cameraPos;
    cameraPos.x = _cameraCenter.x + _distance * cos(_pitch) * sin(_yaw);
    cameraPos.y = _cameraCenter.y + _distance * sin(_pitch);
    cameraPos.z = _cameraCenter.z + _distance * cos(_pitch) * cos(_yaw);

    glm::mat4 projection = glm::perspective(
        glm::radians(DEFAULT_FOV),
        _aspectRatio,
        DEFAULT_NEAR_PLANE,
        DEFAULT_FAR_PLANE
    );

    glm::mat4 view = glm::lookAt(
        cameraPos,
        _cameraCenter,
        glm::vec3(0.0f, 1.0f, 0.0f)
    );

    _cameraMatrix = projection * view;
}




void FrameGL::zoom(float delta) {
    _distance -= delta * ZOOM_STEP;
    if (_distance < MIN_DISTANCE)
        _distance = MIN_DISTANCE;

    updateCameraMatrix();
}

void FrameGL::translateCamera(float deltaX, float deltaY) {
    const float sensitivity = 0.005f * _distance;

    _cameraCenter.x -= deltaX * sensitivity;
    _cameraCenter.y += deltaY * sensitivity;

    updateCameraMatrix();
}

void FrameGL::rotateCamera(float deltaYaw, float deltaPitch) {
    _yaw += deltaYaw;
    _pitch += deltaPitch;
    updateCameraMatrix();
}

void FrameGL::resetCamera() {
    _yaw = 0.0f;
    _pitch = 0.0f;
    _distance = DEFAULT_DISTANCE;
    _cameraCenter = glm::vec3(0.0f, 0.0f, 0.0f);
    updateCameraMatrix();
}




void FrameGL::setRegions(const Regions & regions) {
    _regions = regions;
    size_t numberOfRegions = _regions.size();
    _vaos.clear();
    _numVertices.clear();
    for (size_t i = 0; i < numberOfRegions; i ++) {
        _vaos.push_back(std::make_unique<VAO>(1));
    }
    for (size_t i = 0; i < numberOfRegions; i ++) {
        std::vector<glm::vec3> vertices = _regions[i].createMesh();
        _vaos[i] -> setVector(VERTEX_BUFFER, vertices);
        _numVertices.push_back((unsigned)vertices.size());

        int idRegion = _regions[i].getId();
        if (_tableColor.find(idRegion) == _tableColor.end()) {
            _tableColor[idRegion] = generateRandomColor();
        }
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
        _vaos[i] -> bind();
        int idRegion = _regions[i].getId();
        Color colorRegion = _tableColor[idRegion];
        ColorGL colorGL = ColorGL(colorRegion);
        glm::vec3 color(colorGL.r, colorGL.g, colorGL.b);
        _shader -> setUniformVector("color", color);
        glDrawArrays(GL_TRIANGLES, 0, _numVertices[i]);
    }   
}

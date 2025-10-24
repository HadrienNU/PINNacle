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
    
    _camera.distance = DEFAULT_DISTANCE;
    _camera.aspectRatio = 1.0f;
    _camera.yaw = 0.0f;
    _camera.pitch = 0.0f;    
    _camera.position = glm::vec3(0.0f);
    _camera.lookAt = glm::vec3(0.0f);
    _camera.transform = glm::mat4(1.0f);
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
    _camera.aspectRatio = static_cast<float>(frameSize.width) / frameSize.height;
    updateCamera();
}

void FrameGL::updateCamera() {
    float pitchLimit = glm::radians(PITCH_ANGLE_DEG_LIMIT);
    _camera.pitch = glm::clamp(
        _camera.pitch, 
        -pitchLimit, 
        pitchLimit
    );

    glm::vec3 & position = _camera.position;
    position.x = _camera.distance * cos(_camera.pitch) * sin(_camera.yaw);
    position.y = _camera.distance * sin(_camera.pitch);
    position.z = _camera.distance * cos(_camera.pitch) * cos(_camera.yaw);
    position += _camera.lookAt;

    glm::mat4 projection = glm::perspective(
        glm::radians(DEFAULT_FOV_DEG),
        _camera.aspectRatio,
        DEFAULT_NEAR_PLANE,
        DEFAULT_FAR_PLANE
    );

    glm::mat4 view = glm::lookAt(
        position,
        _camera.lookAt,
        glm::vec3(0.0f, 1.0f, 0.0f)
    );

    _camera.transform = projection * view;
}

void FrameGL::scaleCamera(float delta) {
    _camera.distance -= delta * ZOOM_STEP;
    if (_camera.distance < MIN_DISTANCE) {
        _camera.distance = MIN_DISTANCE;
    }
    updateCamera();
}

void FrameGL::translateCamera(float deltaX, float deltaY) {
    float sensitivity = TRANSLATE_SENSIBILITY * _camera.distance;
    
    glm::vec3 centerGaze(0.0f, 0.0f, 1.0f);
    glm::vec3 cameraGaze = glm::normalize(_camera.lookAt - _camera.position);
    float behind = glm::dot(centerGaze, cameraGaze) > 0 ? -1.0f: 1.0f;
    
    _camera.lookAt.x += deltaX * sensitivity * behind;
    _camera.lookAt.y -= deltaY * sensitivity;
    updateCamera();
}

void FrameGL::rotateCamera(float deltaYaw, float deltaPitch) {
    _camera.yaw += deltaYaw * ROTATION_SPEED;
    _camera.pitch -= deltaPitch * ROTATION_SPEED;
    updateCamera();
}

void FrameGL::resetCamera() {
    _camera.yaw = 0.0f;
    _camera.pitch = 0.0f;
    _camera.distance = DEFAULT_DISTANCE;
    _camera.lookAt = glm::vec3(0.0f);
    updateCamera();
}

void FrameGL::setRegions(const Regions & regions) {
    _regions = regions;    
    _vaos.clear();
    _numVertices.clear();
    size_t numberOfRegions = _regions.size();
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
    resetCamera();
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
    _shader -> setUniformMatrix("cameraMatrix", _camera.transform);
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

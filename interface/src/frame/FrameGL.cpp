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
    _distance = DEFAULT_DISTANCE;
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
    glm::mat4 projection = glm::perspective(
        glm::radians(DEFAULT_FOV),
        _aspectRatio,
        DEFAULT_NEAR_PLANE,
        DEFAULT_FAR_PLANE
    );

    glm::vec3 cameraPos = _cameraCenter + glm::vec3(0.0f, 0.0f, _distance);

    glm::mat4 view = glm::lookAt(
        cameraPos,
        _cameraCenter,                // on regarde vers le centre
        glm::vec3(0.0f, 1.0f, 0.0f)   // up = Y+
    );

    _cameraMatrix = projection * view;
}



void FrameGL::zoom(float delta) {
    // delta positif -> zoom avant, delta négatif -> zoom arrière
    _distance -= delta * ZOOM_STEP;

    // sécurité pour ne pas passer derrière ou à 0
    if (_distance < MIN_DISTANCE)
        _distance = MIN_DISTANCE;

    updateCameraMatrix();
}

void FrameGL::translateCamera(float deltaX, float deltaY) {
    const float sensitivity = 0.005f * _distance; // dépend de la distance pour un effet naturel

    // Translation dans le plan caméra (X/Y)
    _cameraCenter.x -= deltaX * sensitivity;
    _cameraCenter.y += deltaY * sensitivity;

    updateCameraMatrix();
}


void FrameGL::setRegions(const Regions & regions) {
    _regions = regions;
    size_t numberOfRegions = _regions.size();
    resizeTableColor(_tableColor, numberOfRegions);
    _vaos.clear();
    _numVertices.clear();
    for (size_t i = 0; i < numberOfRegions; i ++) {
        _vaos.push_back(std::make_unique<VAO>(1));
    }
    for (size_t i = 0; i < numberOfRegions; i ++) {
        std::vector<glm::vec3> vertices = _regions[i].createMesh();
        _vaos[i] -> setVector(VERTEX_BUFFER, vertices);
        _numVertices.push_back((unsigned)vertices.size());
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
        glDrawArrays(GL_TRIANGLES, 0, _numVertices[i]);
    }   
}

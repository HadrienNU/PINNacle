#include <frame/FrameGL.hpp>
#include <GLFW/glfw3.h>
#include <cmath>


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
uniform float alphaPhase;

out vec4 fragColor;

const float ALPHA_MEAN = 0.7;
const float ALPHA_AMPLITUDE = 0.3;
const float ANIMATION_SPEED = 3.0;

void main() {
    float alpha = ALPHA_MEAN + ALPHA_AMPLITUDE * cos(alphaPhase * ANIMATION_SPEED);
    fragColor = vec4(color, alpha);
}
)glsl";


FrameGL::FrameGL(const Color & backgroundColor) :
_backgroundColor(backgroundColor) {
    _shader = nullptr;    
    _msaaFbo = nullptr;
    _resolveFbo = nullptr;
    
    _camera.distance = DEFAULT_DISTANCE;
    _camera.aspectRatio = 1.0f;
    _camera.yaw = 0.0f;
    _camera.pitch = 0.0f;    
    _camera.position = glm::vec3(0.0f);
    _camera.lookAt = glm::vec3(0.0f);
    _camera.transform = glm::mat4(1.0f);
    _pickedRegionId = -1;
}

FrameGL::~FrameGL() {
    delete _shader;
    delete _msaaFbo;
    delete _resolveFbo;
}

static void glInit() {
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
}

void FrameGL::init(const Size & frameSize) {
    glInit();
    _shader = new Shader(vertexShaderSource, fragmentShaderSource);
    _msaaFbo = new FrameBuffer(8);
    _resolveFbo = new FrameBuffer();
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
    
    std::vector<int> regionIds;
    std::vector<std::vector<glm::vec3>> regionMeshes;
    
    size_t numberOfRegions = _regions.size();
    for (size_t i = 0; i < numberOfRegions; i ++) {
        _vaos.push_back(std::make_unique<VAO>(1));
    }
    
    for (size_t i = 0; i < numberOfRegions; i ++) {
        int idRegion = _regions[i].getId();
        std::vector<glm::vec3> vertices = _regions[i].createMesh();
        
        _vaos[i] -> setVector(VERTEX_BUFFER, vertices);
        _numVertices.push_back((unsigned)vertices.size());

        if (_tableColor.find(idRegion) == _tableColor.end()) {
            _tableColor[idRegion] = generateRandomColor();
        }
        
        regionIds.push_back(idRegion);
        regionMeshes.push_back(vertices);
    }
    
    if (!_regions.empty()) {
        _regionPicker.build(regionIds, regionMeshes);
    }
}

const Region* FrameGL::getRegion(int regionId) const {
    for (const auto& region : _regions) {
        if (region.getId() == regionId) {
            return &region;
        }
    }
    return nullptr;
}

int FrameGL::pickRegion(float screenX, float screenY, const Size & frameSize) {
    float x = (2.0f * screenX) / frameSize.width - 1.0f;
    float y = 1.0f - (2.0f * screenY) / frameSize.height;

    glm::vec4 rayClip = glm::vec4(x, y, -1.0f, 1.0f);

    glm::mat4 projection = glm::perspective(
        glm::radians(DEFAULT_FOV_DEG),
        _camera.aspectRatio,
        DEFAULT_NEAR_PLANE,
        DEFAULT_FAR_PLANE
    );
    glm::vec4 rayEye = glm::inverse(projection) * rayClip;
    rayEye = glm::vec4(rayEye.x, rayEye.y, -1.0f, 0.0f);

    glm::mat4 view = glm::lookAt(
        _camera.position,
        _camera.lookAt,
        glm::vec3(0.0f, 1.0f, 0.0f)
    );
    glm::vec3 rayWorld = glm::vec3(glm::inverse(view) * rayEye);
    glm::vec3 rayOrigin = _camera.position;
    glm::vec3 rayDir = glm::normalize(rayWorld);
    _pickedRegionId = _regionPicker.pick(rayOrigin, rayDir);
    return _pickedRegionId;
}

void FrameGL::render() {
    glClearColor(
        _backgroundColor.r, 
        _backgroundColor.g, 
        _backgroundColor.b, 
        1.0f
    );
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    _shader -> bind();
    _shader -> setUniformMatrix("cameraMatrix", _camera.transform);
    
    float currentTime = glfwGetTime();
    for (size_t i = 0; i < _regions.size(); i ++) {
        _vaos[i] -> bind();
        int idRegion = _regions[i].getId();
        Color colorRegion = _tableColor[idRegion];
        ColorGL colorGL = ColorGL(colorRegion);
        glm::vec3 color(colorGL.r, colorGL.g, colorGL.b);
        _shader -> setUniformVector("color", color);
        float alphaPhase = (idRegion == _pickedRegionId) ? currentTime : 0.0f;
        _shader -> setUniformFloat("alphaPhase", alphaPhase);
        glDrawArrays(GL_TRIANGLES, 0, _numVertices[i]);
    }   
}

void FrameGL::renderRegionsOffscreen(std::vector<std::vector<unsigned char>> & images) {
    std::vector<unsigned char> pixels(WIDTH * HEIGHT * 3);

    _msaaFbo->bind();
    glInit();

    glClearColor(
        _backgroundColor.r,
        _backgroundColor.g,
        _backgroundColor.b,
        1.0f
    );
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    _shader->bind();
    _shader->setUniformMatrix("cameraMatrix", _camera.transform);

    for (size_t i = 0; i < _regions.size(); ++i) {
        _vaos[i]->bind();
        ColorGL c(_tableColor[_regions[i].getId()]);
        _shader->setUniformVector("color", {c.r, c.g, c.b});
        _shader->setUniformFloat("alphaPhase", 0.0f);
        glDrawArrays(GL_TRIANGLES, 0, _numVertices[i]);
    }

    glBindFramebuffer(GL_READ_FRAMEBUFFER, _msaaFbo->id());
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, _resolveFbo->id());
    glBlitFramebuffer(
        0, 0, WIDTH, HEIGHT,
        0, 0, WIDTH, HEIGHT,
        GL_COLOR_BUFFER_BIT,
        GL_LINEAR
    );

    _msaaFbo->unbind();
    _resolveFbo->bind();

    glFinish();
    glReadPixels(
        0, 0,
        WIDTH, HEIGHT,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        pixels.data()
    );

    _resolveFbo->unbind();
    images.push_back(pixels);
}

std::vector<std::vector<unsigned char>> FrameGL::renderEpochsOffscreen(
    const std::vector<Regions> & epochs
) {
    std::vector<std::vector<unsigned char>> images;
    for (Regions regions: epochs) {
        setRegions(regions);
        renderRegionsOffscreen(images);
    }
    return images;
}

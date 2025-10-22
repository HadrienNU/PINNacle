#ifndef __FRAME_GL_HPP__
#define __FRAME_GL_HPP__


#include <opengl/Shader.hpp>
#include <opengl/VAO.hpp>
#include <region/Region.hpp>
#include <memory>

#define DEFAULT_FOV 45.0f
#define DEFAULT_NEAR_PLANE 0.1f
#define DEFAULT_FAR_PLANE 100.0f
#define DEFAULT_DISTANCE 3.0f
#define VERTEX_BUFFER 0
#define MIN_DISTANCE 0.1f 
#define ZOOM_STEP 0.2f


class FrameGL {
public:
    FrameGL(const Color & backgroundColor);
    ~FrameGL();
    void init(const Size & frameSize);
    void resize(const Size & frameSize);
    void setRegions(const Regions & regions);
    void render();
    void zoom(float delta);
    void translateCamera(float deltaX, float deltaY);
    void rotateCamera(float deltaYaw, float deltaPitch);
    void resetCamera();

private:
    void updateCameraMatrix();
    Shader * _shader;    
    glm::mat4 _cameraMatrix;
    ColorGL _backgroundColor;
    Regions _regions;
    TableColor _tableColor;
    std::vector<std::unique_ptr<VAO>> _vaos;
    std::vector<unsigned> _numVertices;
    float _distance;
    float _aspectRatio;
    float _yaw; 
    float _pitch;
    glm::vec3 _cameraCenter;

};

#endif

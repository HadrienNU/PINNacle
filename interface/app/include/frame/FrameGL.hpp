#ifndef __FRAME_GL_HPP__
#define __FRAME_GL_HPP__


#include <opengl/Shader.hpp>
#include <opengl/VAO.hpp>
#include <opengl/FrameBuffer.hpp>
#include <region/Region.hpp>
#include <region/regionPicker/RegionPicker.hpp>
#include <memory>

#define DEFAULT_FOV_DEG 45.0f
#define PITCH_ANGLE_DEG_LIMIT 89.0f

#define DEFAULT_NEAR_PLANE 0.1f
#define DEFAULT_FAR_PLANE 100.0f
#define DEFAULT_DISTANCE 3.0f
#define MIN_DISTANCE 0.1f 

#define VERTEX_BUFFER 0

#define ZOOM_STEP 0.2f
#define TRANSLATE_SENSIBILITY 0.0005f
#define ROTATION_SPEED 0.005f

#define ALPHA_MEAN 0.8f
#define ALPHA_AMPLITUDE 0.2f
#define ANIMATION_SPEED 3.0f


struct Camera {    
    float distance;
    float aspectRatio;
    float yaw; 
    float pitch;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::mat4 transform;
};

class FrameGL {
public:
    FrameGL(const Color & backgroundColor);
    ~FrameGL();
    void init(const Size & frameSize);
    void resize(const Size & frameSize);
    void setRegions(const Regions & regions);
    void render();
    void scaleCamera(float delta);
    void translateCamera(float deltaX, float deltaY);
    void rotateCamera(float deltaYaw, float deltaPitch);
    void resetCamera();
    int pickRegion(float screenX, float screenY, const Size & frameSize);
    const Region* getRegion(int regionId) const;
    void renderRegionsOffscreen(std::vector<std::vector<unsigned char>> & images);
    std::vector<std::vector<unsigned char>> renderEpochsOffscreen(
        const std::vector<Regions> & regions
    );
private:
    void updateCamera();
private:
    Shader * _shader;    
    Regions _regions;
    TableColor _tableColor;   
    ColorGL _backgroundColor;   
    Camera _camera;
    FrameBuffer * _msaaFbo;
    FrameBuffer * _resolveFbo;
    std::vector<std::unique_ptr<VAO>> _vaos;
    std::vector<unsigned> _numVertices;
    RegionPicker _regionPicker;
    int _pickedRegionId;
};

#endif

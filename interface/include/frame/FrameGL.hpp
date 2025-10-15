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


class FrameGL {
public:
    FrameGL(const Color & backgroundColor);
    ~FrameGL();
    void init(const Size & frameSize);
    void resize(const Size & frameSize);
    void setRegions(const Regions & regions);
    void render();
private:
    Shader * _shader;    
    glm::mat4 _cameraMatrix;
    ColorGL _backgroundColor;
    Regions _regions;
    TableColor _tableColor;
    std::vector<std::unique_ptr<VAO>> _vaos;
};

#endif

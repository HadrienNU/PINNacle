#ifndef __FRAME_GL_HPP__
#define __FRAME_GL_HPP__


#include <opengl/Shader.hpp>
#include <opengl/VAO.hpp>
#include <region/Region.hpp>

#define VERTEX_BUFFER 0


class FrameGL {
public:
    FrameGL(const Color & backgroundColor);
    ~FrameGL();
    void init();
    void setRegions(const Regions & regions);
    void render();
private:
    Shader * _shader;
    std::vector<VAO> _vaos;
    ColorGL _backgroundColor;
    Regions _regions;
    TableColor _tableColor;
};

#endif

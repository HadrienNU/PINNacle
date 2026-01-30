#ifndef __FRAME_BUFFER_GL_HPP__
#define __FRAME_BUFFER_GL_HPP__


#include <util.hpp>


#define WIDTH 1920
#define HEIGHT 1080


class FrameBuffer {
public:
    FrameBuffer(int samples = 0);
    ~FrameBuffer();
    void bind();
    void unbind();
    GLuint id() const { return _fbo; }
private:
    GLuint _fbo;
    GLuint _colorTex;
    GLuint _depthRbo;
    GLint _oldViewport[4];
};

#endif

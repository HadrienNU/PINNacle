#ifndef __FRAME_BUFFER_GL_HPP__
#define __FRAME_BUFFER_GL_HPP__


#include <util.hpp>


#define WIDTH 600
#define HEIGHT 360


class FrameBuffer {
public:
    FrameBuffer();
    void bind();
    static void unbind();
    GLuint getFBO() const { return fbo; }

private:
    GLuint fbo;
    GLuint colorTex;
};

#endif

#include <opengl/FrameBuffer.hpp>

FrameBuffer::FrameBuffer(int samples) {

    glGenFramebuffers(1, &_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, _fbo);

    if (samples > 0) {
        // MULTISAMPLE
        glGenTextures(1, &_colorTex);
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, _colorTex);
        glTexImage2DMultisample(
            GL_TEXTURE_2D_MULTISAMPLE,
            samples,
            GL_RGB8,
            WIDTH,
            HEIGHT,
            GL_TRUE
        );

        glFramebufferTexture2D(
            GL_FRAMEBUFFER,
            GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D_MULTISAMPLE,
            _colorTex,
            0
        );
    } else {
        // CLASSIC
        glGenTextures(1, &_colorTex);
        glBindTexture(GL_TEXTURE_2D, _colorTex);
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGB8,
            WIDTH,
            HEIGHT,
            0,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            nullptr
        );

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glFramebufferTexture2D(
            GL_FRAMEBUFFER,
            GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D,
            _colorTex,
            0
        );
    }

    // Depth
    glGenRenderbuffers(1, &_depthRbo);
    glBindRenderbuffer(GL_RENDERBUFFER, _depthRbo);

    if (samples > 0) {
        glRenderbufferStorageMultisample(
            GL_RENDERBUFFER,
            samples,
            GL_DEPTH24_STENCIL8,
            WIDTH,
            HEIGHT
        );
    } else {
        glRenderbufferStorage(
            GL_RENDERBUFFER,
            GL_DEPTH24_STENCIL8,
            WIDTH,
            HEIGHT
        );
    }

    glFramebufferRenderbuffer(
        GL_FRAMEBUFFER,
        GL_DEPTH_STENCIL_ATTACHMENT,
        GL_RENDERBUFFER,
        _depthRbo
    );

    assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

FrameBuffer::~FrameBuffer() {
    glDeleteFramebuffers(1, &_fbo);
    glDeleteTextures(1, &_colorTex);
    glDeleteRenderbuffers(1, &_depthRbo);
}

void FrameBuffer::bind() {
    glGetIntegerv(GL_VIEWPORT, _oldViewport);
    glBindFramebuffer(GL_FRAMEBUFFER, _fbo);
    glViewport(0, 0, WIDTH, HEIGHT);
}

void FrameBuffer::unbind() {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(
        _oldViewport[0],
        _oldViewport[1],
        _oldViewport[2],
        _oldViewport[3]
    );
}





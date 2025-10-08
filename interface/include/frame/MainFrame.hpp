#ifndef __MAIN_FRAME_HPP__
#define __MAIN_FRAME_HPP__


#include <GLFW/glfw3.h>
#include <util.hpp>

#if defined(_WIN32)
#  define NOMINMAX
#  include <windows.h>
#  include <GL/gl.h>
#elif defined(__APPLE__)
# ifndef GL_SILENCE_DEPRECATED
#   define GL_SILENCE_DEPRECATED 
#  endif
#  include <OpenGL/gl3.h>
#else
#  include <GL/gl.h>
#endif

class MainFrame {
public:
    MainFrame(const String & title, const Size & frameSize, const Color & backgroundColor);
    ~MainFrame();    
    void run();
private:
    void init();
private:
    String _title;
    Size _frameSize;
    Color _backgroundColor;
    GLFWwindow * _window;
};

#endif

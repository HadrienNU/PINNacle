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
    MainFrame(const String & title, const Size & frameSize);
    ~MainFrame();    
    void run();
private:
    void init();
private:
    String _title;
    Size _frameSize;
    GLFWwindow * _window;
};

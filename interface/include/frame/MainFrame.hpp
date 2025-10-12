#ifndef __MAIN_FRAME_HPP__
#define __MAIN_FRAME_HPP__


#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <util.hpp>
#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_glfw.h>
#include <imgui/backends/imgui_impl_opengl3.h>


class MainFrame {
public:
    MainFrame(const String & title, const Size & frameSize, const Color & backgroundColor);
    ~MainFrame();    
    void run();
private:
    void init();
    void initImGUI();
    void runImGui();
    void runOpenGL();
private:
    String _title;
    Size _frameSize;
    Color _backgroundColor;
    GLFWwindow * _window;
};

#endif

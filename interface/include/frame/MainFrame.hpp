#ifndef __MAIN_FRAME_HPP__
#define __MAIN_FRAME_HPP__


#include <GLFW/glfw3.h>
#include <util.hpp>
#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_glfw.h>
#include <imgui/backends/imgui_impl_opengl3.h>
#include <frame/FrameGL.hpp>
#include <frame/ImGuiFrames.hpp>


class MainFrame {
public:
    MainFrame(
        const String & title, 
        const Size & frameSize,
        FrameGL * frameGL,
        ImGuiFrames & imguiFrames
    );
    ~MainFrame();    
    void resize(const Size & frameSize);
    void run();
private:
    void init();
    void initImGUI();
private:
    String _title;
    Size _frameSize;
    GLFWwindow * _window;
    FrameGL * _frameGL;
    ImGuiFrames _imguiFrames;
};

#endif

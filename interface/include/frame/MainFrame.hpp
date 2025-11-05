#ifndef __MAIN_FRAME_HPP__
#define __MAIN_FRAME_HPP__


#include <GLFW/glfw3.h>
#include <util.hpp>
#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_glfw.h>
#include <imgui/backends/imgui_impl_opengl3.h>
#include <frame/FrameGL.hpp>
#include <frame/FramesImGui.hpp>


struct Event {
    double mousePositionX;
    double mousePositionY;
    bool rightButtonPressed;
    bool leftButtonPressed;
};

class MainFrame {
public:
    MainFrame(
        const String & title, 
        const Size & frameSize,
        FrameGL * frameGL,
        FramesImGui & imguiFrames
    );
    ~MainFrame();
    Event & event();    
    void resize(const Size & frameSize);
    void scaleCamera(float delta);
    void translateCamera(float deltaX, float deltaY);
    void rotateCamera(float deltaYaw, float deltaPitch);
    void run();
private:
    void init();
    void initImGUI();
private:
    String _title;
    Size _frameSize;    
    Event _event;
    GLFWwindow * _window;
    FrameGL * _frameGL;
    FramesImGui & _imguiFrames;
};

#endif

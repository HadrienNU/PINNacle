#include <frame/ImGuiFrames.hpp>
#include <imgui/backends/imgui_impl_opengl3.h>
#include <imgui/backends/imgui_impl_glfw.h>


ImGuiFrames::ImGuiFrames(FrameGL * frameGL) {
    std::shared_ptr<FrameInfo> frameInfo = std::make_shared<FrameInfo>();
    _imguiFrames.push_back(frameInfo);
    _imguiFrames.push_back(std::make_shared<FrameFileSelection>(frameGL, frameInfo));
}

void ImGuiFrames::render() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    for (std::shared_ptr<ImGuiFrame>& frame : _imguiFrames) {
        if (frame && frame->isVisible()) {
            frame->render();
        }
    }

    ImGui::Render();
}

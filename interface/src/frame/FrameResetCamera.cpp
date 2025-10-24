#include <frame/FrameResetCamera.hpp>


FrameResetCamera::FrameResetCamera(FrameGL * frameGL) 
    : FrameImGui("##ResetCamera", true), _frameGL(frameGL) {}

void FrameResetCamera::render() {
    if (!_isVisible) {
        return;
    }

    ImGuiIO & io = ImGui::GetIO();
    ImVec2 windowSize(RESET_CAMERA_BUTTON_WIDTH, RESET_CAMERA_BUTTON_HEIGHT);
    ImVec2 windowPos(
        io.DisplaySize.x - windowSize.x,
        io.DisplaySize.y - windowSize.y
    );

    ImGui::SetNextWindowPos(windowPos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(windowSize, ImGuiCond_Always);

    ImGui::Begin(
        _title.c_str(), 
        nullptr,
        ImGuiWindowFlags_NoResize | 
        ImGuiWindowFlags_NoMove | 
        ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoBackground
    );

    if (ImGui::Button(RESET_CAMERA_BUTTON_TEXT, ImVec2(-1, -1))) {
        _frameGL->resetCamera();
    }

    ImGui::End();
}

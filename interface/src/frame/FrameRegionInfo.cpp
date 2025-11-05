#include <frame/FrameRegionInfo.hpp>
#include <GLFW/glfw3.h>


FrameRegionInfo::FrameRegionInfo(FrameGL * frameGL) 
    : FrameImGui(REGION_INFO_DEFAULT_TITLE, false), _region(-1), _frameGL(frameGL) {}

void FrameRegionInfo::render() {
    if (!_isVisible) return;

    ImGuiIO & io = ImGui::GetIO();
    float scale = io.FontGlobalScale;
    ImVec2 windowSize(REGION_INFO_WINDOW_WIDTH * scale, REGION_INFO_WINDOW_HEIGHT * scale);
    
    float posX = io.DisplaySize.x - windowSize.x;
    float posY = INFO_WINDOW_HEIGHT * scale;
    
    ImGui::SetNextWindowSize(windowSize, ImGuiCond_Always);
    ImGui::SetNextWindowPos(ImVec2(posX, posY), ImGuiCond_Always);
    
    ImGui::Begin(_title.c_str(), &_isVisible);
    
    ImGui::TextColored(IMGUI_TITLE_COLOR, REGION_INFO_TITLE);
    ImGui::Separator();
    ImGui::Spacing();
    
    ImGui::Text("%s %d", REGION_INFO_LABEL, _region.getId());
    
    ImGui::End();
}



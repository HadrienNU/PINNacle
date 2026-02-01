#include <frame/FrameRegionInfo.hpp>
#include <GLFW/glfw3.h>
#include <iostream>
#include <iostream>


FrameRegionInfo::FrameRegionInfo(FrameGL * frameGL) 
    : FrameImGui(REGION_INFO_DEFAULT_TITLE, false), _region(-1), _frameGL(frameGL) {}

void FrameRegionInfo::showRegionInfo(int regionId) {
    if (regionId >= 0) {
        const Region* loadedRegion = _frameGL->getRegion(regionId);
        if (loadedRegion) {
            _region = *loadedRegion;
        } else {
            _region = Region(regionId);
        }
        show();
    }
}

void FrameRegionInfo::hideRegionInfo() {
    _region = Region(-1);
    hide();
}

void FrameRegionInfo::render() {
    if (!_isVisible) return;

    ImGuiIO & io = ImGui::GetIO();
    float scale = io.FontGlobalScale;
    
    auto stats = _region.getStatistics();  
    
    float totalHeight = REGION_INFO_WINDOW_HEIGHT;
    if (!stats.empty()) {
        totalHeight += (stats.size() * REGION_INFO_STAT_LINE_HEIGHT);
    }
    
    ImVec2 windowSize(REGION_INFO_WINDOW_WIDTH * scale, totalHeight * scale);
    
    float posX = io.DisplaySize.x - windowSize.x;
    float posY = INFO_WINDOW_HEIGHT * scale;
    
    ImGui::SetNextWindowSize(windowSize, ImGuiCond_Always);
    ImGui::SetNextWindowPos(ImVec2(posX, posY), ImGuiCond_Always);
    
    ImGui::Begin(
        _title.c_str(), 
        &_isVisible,
        ImGuiWindowFlags_NoDecoration |
        ImGuiWindowFlags_NoMove 
    );

    ImGui::TextColored(IMGUI_TITLE_COLOR, REGION_INFO_TITLE);
    ImGui::Separator();
    ImGui::Spacing();
    
    ImGui::Text("%s %d", REGION_INFO_LABEL, _region.getId());
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    
    if (!stats.empty()) {
        ImGui::Spacing();
        
        for (const auto& stat : stats) {
            ImGui::Text("%s: %.6f", stat.first.c_str(), stat.second);
        }
    } else {
        ImGui::TextDisabled(REGION_INFO_NO_STATISTICS);
    }
    
    ImGui::End();
}



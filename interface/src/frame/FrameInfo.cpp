#include <frame/FrameInfo.hpp>


FrameInfo::FrameInfo()
    : FrameImGui(INFO_DEFAULT_TITLE, true),
      _regionCount(INFO_DEFAULT_REGION_COUNT),
      _currentFile(INFO_NO_FILE_LOADED) {}

void FrameInfo::render() {
    if (!_isVisible) {
        return;
    }

    ImGuiIO & io = ImGui::GetIO();
    float scale = io.FontGlobalScale;
    ImVec2 windowSize(INFO_WINDOW_WIDTH * scale, INFO_WINDOW_HEIGHT * scale);
    ImGui::SetNextWindowSize(windowSize, ImGuiCond_Always);
    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - windowSize.x, 0), ImGuiCond_Always);
    
    ImGui::Begin(_title.c_str(), &_isVisible);

    ImGui::TextColored(IMGUI_TITLE_COLOR, INFO_TITLE);
    ImGui::Separator();

    ImGui::Text(INFO_CURRENT_FILE_LABEL);
    ImGui::Indent();
    ImGui::Text("%s", _currentFile.c_str());
    ImGui::Unindent();
    ImGui::Separator();

    ImGui::Text(INFO_STATISTICS_LABEL);
    ImGui::Indent();
    ImGui::Text("%s %zu", INFO_REGION_COUNT_LABEL, _regionCount);
    ImGui::Unindent();

    ImGui::End();
}

void FrameInfo::setRegionCount(size_t regionCount) {
    _regionCount = regionCount;
}

void FrameInfo::setCurrentFile(const String& filename) {
    _currentFile = filename;
}

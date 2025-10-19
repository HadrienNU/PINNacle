#include <frame/FrameInfo.hpp>


FrameInfo::FrameInfo(const String & title, bool visible)
    : ImGuiFrame(title, visible),
      _regionCount(FRAME_INFO_NO_REGIONS_LOADED),
      _currentFile(FRAME_INFO_NO_FILE_LOADED) {}

void FrameInfo::render() {
    if (!_isVisible) {
        return;
    }

    ImGuiIO & io = ImGui::GetIO();
    ImVec2 windowSize(300, 125);
    ImGui::SetNextWindowSize(windowSize, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - windowSize.x, 0), ImGuiCond_Always);
    
    ImGui::Begin(_title.c_str(), &_isVisible);

    ImGui::TextColored(ImVec4(0.2f, 0.8f, 1.0f, 1.0f), FRAME_INFO_TITLE);
    ImGui::Separator();

    ImGui::Text(FRAME_INFO_CURRENT_FILE);
    ImGui::Indent();
    ImGui::Text("%s", _currentFile.c_str());
    ImGui::Unindent();
    ImGui::Separator();

    ImGui::Text(FRAME_INFO_STATISTICS);
    ImGui::Indent();
    ImGui::Text("%s %zu", FRAME_INFO_REGION_COUNT, _regionCount);
    ImGui::Unindent();

    ImGui::End();
}

void FrameInfo::setRegionCount(size_t regionCount) {
    _regionCount = regionCount;
}

void FrameInfo::setCurrentFile(const String& filename) {
    _currentFile = filename;
}

#include <frame/FrameSettings.hpp>


FrameSettings::FrameSettings(FrameGL * frameGL) 
    : FrameImGui("##Settings", true), 
      _frameGL(frameGL), 
      _settingsWindowOpen(false), 
      _fontScale(DEFAULT_FONT_SCALE) {}

void FrameSettings::render() {
    if (!_isVisible) {
        return;
    }

    ImGuiIO & io = ImGui::GetIO();
    float scale = io.FontGlobalScale;
    ImVec2 buttonSize(SETTINGS_BUTTON_WIDTH * scale, SETTINGS_BUTTON_HEIGHT * scale);
    ImVec2 buttonPos(0, io.DisplaySize.y - buttonSize.y);
    ImGui::SetNextWindowPos(buttonPos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(buttonSize, ImGuiCond_Always);

    ImGui::Begin(
        _title.c_str(), 
        nullptr,
        ImGuiWindowFlags_NoMove | 
        ImGuiWindowFlags_NoDecoration |
        ImGuiWindowFlags_NoBackground
    );

    if (ImGui::Button(SETTINGS_BUTTON_TEXT, ImVec2(-1, -1))) {
        _settingsWindowOpen = !_settingsWindowOpen;
    }

    ImGui::End();

    if (_settingsWindowOpen) {
        ImVec2 windowSize(SETTINGS_WINDOW_WIDTH * scale, SETTINGS_WINDOW_HEIGHT * scale);
        ImVec2 windowPos(
            (io.DisplaySize.x - windowSize.x) * SETTINGS_WINDOW_CENTER_X,
            (io.DisplaySize.y - windowSize.y) * SETTINGS_WINDOW_CENTER_Y
        );

        ImGui::SetNextWindowPos(windowPos, ImGuiCond_Always);
        ImGui::SetNextWindowSize(windowSize, ImGuiCond_Always);

        ImGui::Begin(
            SETTINGS_WINDOW_TITLE, 
            &_settingsWindowOpen, 
            ImGuiWindowFlags_NoCollapse |
            ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove 
        );

        ImGui::TextColored(IMGUI_TITLE_COLOR, SETTINGS_FONT_SECTION_TITLE);
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::Text(SETTINGS_FONT_SCALE_LABEL);
        ImGui::Spacing();
    
        float buttonWidth = SETTINGS_RESET_BUTTON_WIDTH * scale;
        float spacing = ImGui::GetStyle().ItemSpacing.x;
        float availableWidth = ImGui::GetContentRegionAvail().x;
        float sliderWidth = availableWidth - buttonWidth - spacing;
        
        ImGui::SetNextItemWidth(sliderWidth);
        if (ImGui::SliderFloat("##FontScale", &_fontScale, MIN_FONT_SCALE, MAX_FONT_SCALE, SETTINGS_FONT_SCALE_FORMAT)) {
            io.FontGlobalScale = _fontScale;
        }
        ImGui::SameLine();
        if (ImGui::Button(SETTINGS_RESET_BUTTON_TEXT, ImVec2(buttonWidth, 0))) {
            _fontScale = DEFAULT_FONT_SCALE;
            io.FontGlobalScale = DEFAULT_FONT_SCALE;
        }

        ImGui::End();
    }
}

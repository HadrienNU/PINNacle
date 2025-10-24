#ifndef __FRAME_SETTINGS_HPP__
#define __FRAME_SETTINGS_HPP__


#include <frame/FrameImGui.hpp>
#include <frame/FrameGL.hpp>

#define SETTINGS_BUTTON_WIDTH 120.0f
#define SETTINGS_BUTTON_HEIGHT 50.0f
#define SETTINGS_BUTTON_TEXT "Settings"

#define SETTINGS_WINDOW_WIDTH 350.0f
#define SETTINGS_WINDOW_HEIGHT 150.0f

#define SETTINGS_WINDOW_CENTER_X 0.5f
#define SETTINGS_WINDOW_CENTER_Y 0.5f

#define SETTINGS_WINDOW_TITLE "Settings"

#define SETTINGS_FONT_SECTION_TITLE "Font Settings"
#define SETTINGS_FONT_SCALE_LABEL "Font Scale:"
#define SETTINGS_FONT_SCALE_FORMAT "%.2fx"

#define SETTINGS_RESET_BUTTON_TEXT "Reset"
#define SETTINGS_RESET_BUTTON_WIDTH 80.0f

#define DEFAULT_FONT_SCALE 1.0f
#define MIN_FONT_SCALE 0.5f
#define MAX_FONT_SCALE 3.0f


class FrameSettings : public FrameImGui {
public:
    FrameSettings(FrameGL * frameGL);
    ~FrameSettings() = default;
    void render() override;
    float getFontScale() const { return _fontScale; }
private:
    FrameGL * _frameGL;
    bool _settingsWindowOpen;
    float _fontScale;
};

#endif

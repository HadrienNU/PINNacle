#ifndef __FRAME_IMGUI_HPP__
#define __FRAME_IMGUI_HPP__


#include <imgui/imgui.h>
#include <util.hpp>

#define FRAME_IMGUI_TITLE_COLOR ImVec4(0.2f, 0.8f, 1.0f, 1.0f)


class FrameImGui {
public:
    FrameImGui(const String & title, bool visible = true) 
            : _title(title), _isVisible(visible) {}
    virtual ~FrameImGui() = default;
    virtual void render() = 0;
    virtual bool isVisible() const { return _isVisible; }
protected:
    String _title;
    bool _isVisible;
};

#endif

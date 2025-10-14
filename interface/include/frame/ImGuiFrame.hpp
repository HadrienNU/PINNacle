#ifndef __IMGUI_FRAME_HPP__
#define __IMGUI_FRAME_HPP__


#include <imgui/imgui.h>
#include <util.hpp>


class ImGuiFrame {
public:
    ImGuiFrame(const String& title, bool visible = true) 
            : _title(title), _isVisible(visible) {}
    virtual ~ImGuiFrame() = default;
    virtual void render() = 0;
    virtual bool isVisible() const { return _isVisible; }
protected:
    String _title;
    bool _isVisible;
};

#endif

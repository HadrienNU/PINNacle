#ifndef __IMGUI_FRAMES_HPP__
#define __IMGUI_FRAMES_HPP__


#include <frame/FrameGL.hpp>
#include <frame/ImGuiFrame.hpp>
#include <frame/FrameInfo.hpp>
#include <frame/FrameFileSelection.hpp>
#include <vector>
#include <memory>


class ImGuiFrames {
public:
    ImGuiFrames(FrameGL * frameGL);
    ~ImGuiFrames() = default;
    void render();
private:
    std::vector<std::shared_ptr<ImGuiFrame>> _imguiFrames;
};

#endif

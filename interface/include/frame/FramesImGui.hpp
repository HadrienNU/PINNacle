#ifndef __FRAMES_IMGUI_HPP__
#define __FRAMES_IMGUI_HPP__


#include <frame/FrameGL.hpp>
#include <frame/FrameImGui.hpp>
#include <frame/FrameInfo.hpp>
#include <frame/FrameFileSelection.hpp>
#include <frame/FrameCenterCamera.hpp>
#include <frame/FrameSettings.hpp>
#include <frame/FrameRegionInfo.hpp>
#include <vector>
#include <memory>


class FramesImGui {
public:
    FramesImGui(FrameGL * frameGL);
    ~FramesImGui() = default;
    void render();
    void showAll();
    void hideAll();
    std::shared_ptr<FrameRegionInfo> getRegionInfoFrame() const { return _regionInfoFrame; }
private:
    std::vector<std::shared_ptr<FrameImGui>> _imguiFrames;
    std::shared_ptr<FrameRegionInfo> _regionInfoFrame;
};

#endif

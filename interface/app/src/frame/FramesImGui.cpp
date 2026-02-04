#include <frame/FramesImGui.hpp>
#include <imgui/backends/imgui_impl_opengl3.h>
#include <imgui/backends/imgui_impl_glfw.h>


FramesImGui::FramesImGui(FrameGL * frameGL) {
    std::shared_ptr<FrameInfo> frameInfo = std::make_shared<FrameInfo>();
    _imguiFrames.push_back(frameInfo);
    
    std::shared_ptr<FrameFileSelection> fileSelection = std::make_shared<FrameFileSelection>(frameGL, frameInfo);
    _imguiFrames.push_back(fileSelection);
    
    _imguiFrames.push_back(std::make_shared<FrameRecordAnimation>(
        frameGL,
        [fileSelection]() { return fileSelection->getBinFiles(); },
        [fileSelection]() { return fileSelection->getCurrentFolderPath(); }
    ));
    
    _imguiFrames.push_back(std::make_shared<FrameResetCamera>(frameGL));
    _imguiFrames.push_back(std::make_shared<FrameSettings>(frameGL));
    _regionInfoFrame = std::make_shared<FrameRegionInfo>(frameGL);
    _imguiFrames.push_back(_regionInfoFrame);
}

void FramesImGui::render() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    for (std::shared_ptr<FrameImGui>& frame : _imguiFrames) {
        if (frame && frame->isVisible()) {
            frame->render();
        }
    }

    ImGui::Render();
}

void FramesImGui::showAll() {
    for (std::shared_ptr<FrameImGui>& frame : _imguiFrames) {
        if (frame) {
            frame->show();
        }
    }
}

void FramesImGui::hideAll() {
    for (std::shared_ptr<FrameImGui>& frame : _imguiFrames) {
        if (frame) {
            frame->hide();
        }
    }
}
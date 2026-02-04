#ifndef __FRAME_RECORD_ANIMATION_HPP__
#define __FRAME_RECORD_ANIMATION_HPP__


#include <frame/FrameImGui.hpp>
#include <frame/FrameGL.hpp>
#include <region/RegionReader.hpp>
#include <thread>
#include <atomic>
#include <functional>

#define RECORD_ANIMATION_BUTTON_WIDTH 120.0f
#define RECORD_ANIMATION_BUTTON_HEIGHT 50.0f
#define RECORD_ANIMATION_BUTTON_TEXT "Record"

#define RECORD_ANIMATION_OFFSET_Y 60.0f


class FrameRecordAnimation : public FrameImGui {
public:
    FrameRecordAnimation(FrameGL* frameGL, std::function<std::vector<String>()> getBinFiles, std::function<String()> getCurrentFolderPath);
    ~FrameRecordAnimation() = default;
    void render() override;
private:
    void createAnimation();
private:
    FrameGL* _frameGL;
    RegionReader _regionReader;
    std::function<std::vector<String>()> _getBinFiles;
    std::function<String()> _getCurrentFolderPath;
};

#endif

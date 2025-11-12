#ifndef __FRAME_REGION_INFO_HPP__
#define __FRAME_REGION_INFO_HPP__


#include <frame/FrameImGui.hpp>
#include <frame/FrameGL.hpp>
#include <frame/FrameInfo.hpp>
#include <region/Region.hpp>

#define REGION_INFO_DEFAULT_TITLE "Region Info"
#define REGION_INFO_TITLE "Region Information"
#define REGION_INFO_LABEL "Region ID:"

#define REGION_INFO_WINDOW_WIDTH 300.0f
#define REGION_INFO_WINDOW_HEIGHT 70.0f


class FrameRegionInfo : public FrameImGui {
public:
    FrameRegionInfo(FrameGL * frameGL);
    ~FrameRegionInfo() = default;
    void render() override;
    void showRegionInfo(int regionId);
    void hideRegionInfo();
private:
    Region _region;
    FrameGL * _frameGL;
};

#endif

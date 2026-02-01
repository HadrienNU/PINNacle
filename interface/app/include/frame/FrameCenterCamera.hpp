#ifndef __FRAME_RESET_CAMERA_HPP__
#define __FRAME_RESET_CAMERA_HPP__


#include <frame/FrameImGui.hpp>
#include <frame/FrameGL.hpp>

#define RESET_CAMERA_BUTTON_WIDTH 120.0f
#define RESET_CAMERA_BUTTON_HEIGHT 50.0f
#define RESET_CAMERA_BUTTON_TEXT "Center"


class FrameResetCamera : public FrameImGui {
public:
    FrameResetCamera(FrameGL * frameGL);
    ~FrameResetCamera() = default;
    void render() override;
private:
    FrameGL * _frameGL;
};

#endif

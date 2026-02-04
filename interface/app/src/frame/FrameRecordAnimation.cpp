#include <frame/FrameRecordAnimation.hpp>
#include <chrono>
#include <iostream>
#include <cstdio>

#ifdef _WIN32
    #include <io.h>
    #include <fcntl.h>
    #define popen  _popen
    #define pclose _pclose
#endif

#define WIDTH 1920
#define HEIGHT 1080


FrameRecordAnimation::FrameRecordAnimation(FrameGL* frameGL, std::function<std::vector<String>()> getBinFiles, std::function<String()> getCurrentFolderPath)
    : FrameImGui("##RecordAnimation", true),
      _frameGL(frameGL),
      _getBinFiles(getBinFiles),
      _getCurrentFolderPath(getCurrentFolderPath) {}

void FrameRecordAnimation::render() {
    if (!_isVisible) {
        return;
    }

    ImGuiIO & io = ImGui::GetIO();
    float scale = io.FontGlobalScale;
    ImVec2 windowSize(RECORD_ANIMATION_BUTTON_WIDTH * scale, RECORD_ANIMATION_BUTTON_HEIGHT * scale);
    ImVec2 windowPos(
        io.DisplaySize.x - windowSize.x,
        io.DisplaySize.y - windowSize.y - (RECORD_ANIMATION_OFFSET_Y * scale)
    );

    ImGui::SetNextWindowPos(windowPos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(windowSize, ImGuiCond_Always);

    ImGui::Begin(
        _title.c_str(), 
        nullptr, 
        ImGuiWindowFlags_NoMove | 
        ImGuiWindowFlags_NoDecoration |
        ImGuiWindowFlags_NoBackground
    );

    bool hasFiles = !_getBinFiles().empty();
    ImGui::BeginDisabled(!hasFiles);
    if (ImGui::Button(RECORD_ANIMATION_BUTTON_TEXT, ImVec2(-1, -1))) {
        createAnimation();
    }
    ImGui::EndDisabled();

    ImGui::End();
}

void FrameRecordAnimation::createAnimation() {
    std::vector<String> binFiles = _getBinFiles();
    if (binFiles.empty()) {
        return;
    }
    
    std::vector<Regions> epochs;
    String folderPath = _getCurrentFolderPath();
    
    for (size_t i = 0; i < binFiles.size(); i++) {
        String filename = binFiles[i];
        String filepath = folderPath + filename;
        _regionReader.setRegionFilePath(filepath);
        Regions regions = _regionReader.read();
        epochs.push_back(regions);
    }
    
    std::vector<std::vector<unsigned char>> images;
    images = _frameGL->renderEpochsOffscreen(epochs);
    if (images.empty()) {
        return;
    }

    String outputPath = folderPath + "animation.mp4";
    String ffmpegCmd = "ffmpeg -y -f rawvideo -pixel_format rgb24 -video_size 1920x1080 -framerate 5 -i - -c:v libx264 -pix_fmt yuv420p \"" + outputPath + "\"";

    FILE* pipe = popen(ffmpegCmd.c_str(), "wb");
    if (!pipe) {
        std::cerr << "Failed to open ffmpeg pipe\n";
        return;
    }

    const size_t frameSize = WIDTH * HEIGHT * 3;
    for (const auto& frame : images) {
        if (frame.size() != frameSize) {
            continue;
        }

        std::vector<unsigned char> flipped(frameSize);

        for (int y = 0; y < HEIGHT; ++y) {
            memcpy(
                &flipped[y * WIDTH * 3],
                &frame[(HEIGHT - 1 - y) * WIDTH * 3],
                WIDTH * 3
            );
        }

        fwrite(flipped.data(), 1, frameSize, pipe);
    }
    pclose(pipe);
}
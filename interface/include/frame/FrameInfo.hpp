#ifndef __INFO_FRAME_HPP__
#define __INFO_FRAME_HPP__


#include <frame/FrameImGui.hpp>

#define FRAME_INFO_DEFAULT_TITLE "Information"
#define FRAME_INFO_NO_FILE_LOADED "No files loaded"
#define FRAME_INFO_NO_REGIONS_LOADED 0
#define FRAME_INFO_TITLE "PINNacle - Visualization Interface"
#define FRAME_INFO_CURRENT_FILE "Current file:"
#define FRAME_INFO_STATISTICS "Statistics:"
#define FRAME_INFO_REGION_COUNT "Number of regions:"

#define FRAME_INFO_WIDTH 300
#define FRAME_INFO_HEIGHT 125


class FrameInfo : public FrameImGui {
public:
    FrameInfo();
    void render() override;
    void setRegionCount(size_t regionCount);
    void setCurrentFile(const String & filename);
private:
    size_t _regionCount;
    String _currentFile;
};

#endif

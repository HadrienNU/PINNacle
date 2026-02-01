#ifndef __INFO_FRAME_HPP__
#define __INFO_FRAME_HPP__


#include <frame/FrameImGui.hpp>

#define INFO_DEFAULT_TITLE "Information"
#define INFO_TITLE "PINNacle - Visualization Interface"

#define INFO_CURRENT_FILE_LABEL "Current file:"
#define INFO_NO_FILE_LOADED "No files loaded"

#define INFO_STATISTICS_LABEL "Statistics:"
#define INFO_REGION_COUNT_LABEL "Number of regions:"
#define INFO_DEFAULT_REGION_COUNT 0

#define INFO_WINDOW_WIDTH 300.0f
#define INFO_WINDOW_HEIGHT 125.0f


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

#ifndef __FILE_SELECTION_FRAME_HPP__
#define __FILE_SELECTION_FRAME_HPP__


#include <frame/ImGuiFrame.hpp>
#include <frame/FrameGL.hpp>
#include <region/RegionReader.hpp>
#include <frame/FrameInfo.hpp>
#include <memory>
#include <filesystem>

#define FRAME_FILE_SELECTION_DEFAULT_TITLE "File Selection"
#define FRAME_FILE_SELECTION_RUNS_PATH "../runs/"
#define FRAME_FILE_SELECTION_TITLE "Select CSV File"
#define FRAME_FILE_SELECTION_REFRESH_BUTTON "Refresh File List"
#define FRAME_FILE_SELECTION_LABEL "CSV File:"
#define FRAME_FILE_SELECTION_NO_FILE "No file selected"


class FrameFileSelection : public ImGuiFrame {
public:
    FrameFileSelection(FrameGL * frameGL, std::shared_ptr<FrameInfo> frameInfo, const String & title = FRAME_FILE_SELECTION_DEFAULT_TITLE, bool visible = true);
    void render() override;
private:
    void loadFile(const String & filename);
    void scanCSVFiles();
    size_t countRegions(const String & filepath);
private:
    std::shared_ptr<FrameInfo> _frameInfo;
    std::vector<String> _csvFiles;
    int _selectedFileIndex;
    RegionReader _regionReader;
    FrameGL * _frameGL;
};

#endif

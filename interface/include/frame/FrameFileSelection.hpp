#ifndef __FILE_SELECTION_FRAME_HPP__
#define __FILE_SELECTION_FRAME_HPP__


#include <frame/FrameImGui.hpp>
#include <frame/FrameGL.hpp>
#include <region/RegionReader.hpp>
#include <frame/FrameInfo.hpp>
#include <memory>
#include <filesystem>
#include <functional>

#define FRAME_FILE_SELECTION_DEFAULT_TITLE "File Selection"
#define FRAME_FILE_SELECTION_RUNS_PATH "../runs/"
#define FRAME_FILE_SELECTION_FOLDER_PREFIX "runs/"
#define FRAME_FILE_SELECTION_TITLE "Select CSV File"
#define FRAME_FILE_SELECTION_LABEL "CSV File:"
#define FRAME_FILE_SELECTION_NO_FILE "No file selected"
#define FRAME_FILE_SELECTION_FOLDER_LABEL "Folder:"
#define FRAME_FILE_SELECTION_NO_FOLDER "No folder selected"

#define FRAME_FILE_SELECTION_WIDTH 300
#define FRAME_FILE_SELECTION_HEIGHT 140

#define FRAME_FILE_SELECTION_NO_INDEX -1
#define FRAME_FILE_SELECTION_DEFAULT_FOLDER_INDEX 0


class FrameFileSelection : public FrameImGui {
public:
    FrameFileSelection(FrameGL * frameGL, std::shared_ptr<FrameInfo> frameInfo);
    void render() override;
private:
    void loadFile(const String & filename);
    void scanCSVFiles();
    void scanFolders();
    void scanDirectory(const String & path, std::vector<String> & results, bool filesOnly, const String & extension = "");
    void renderComboBox(const char * label, const char * comboId, const std::vector<String> & items, int & selectedIndex, const char * noSelectionText, std::function<void(int)> onSelectionChanged = nullptr);
    String getCurrentFolderPath() const;
    size_t countRegions(const String & filepath);
    void selectPreviousFile();
    void selectNextFile();
private:
    std::shared_ptr<FrameInfo> _frameInfo;
    std::vector<String> _csvFiles;
    std::vector<String> _folders;
    int _selectedFileIndex;
    int _selectedFolderIndex;
    RegionReader _regionReader;
    FrameGL * _frameGL;
};

#endif

#ifndef __FILE_SELECTION_FRAME_HPP__
#define __FILE_SELECTION_FRAME_HPP__


#include <frame/FrameImGui.hpp>
#include <frame/FrameGL.hpp>
#include <region/RegionReader.hpp>
#include <frame/FrameInfo.hpp>
#include <frame/fileSelection/FileObserver.hpp>
#include <memory>
#include <filesystem>
#include <functional>
#include <thread>
#include <atomic>
#include <cstdio>
#include <iostream>

#define FILE_SELECTION_DEFAULT_TITLE "File Selection"
#define FILE_SELECTION_TITLE "Select BIN File"

#define FILE_SELECTION_FOLDER_LABEL "Folder:"
#define FILE_SELECTION_NO_FOLDER "No folder selected"

#define FILE_SELECTION_FILE_LABEL "BIN File:"
#define FILE_SELECTION_NO_FILE "No file selected"

#define FILE_SELECTION_RUNS_PATH "../runs/"
#define FILE_SELECTION_FOLDER_PREFIX "runs/"

#define FILE_SELECTION_WINDOW_WIDTH 300.0f
#define FILE_SELECTION_WINDOW_HEIGHT 190.0f

#define FILE_SELECTION_NO_INDEX -1
#define FILE_SELECTION_DEFAULT_FOLDER_INDEX 0
#define FILE_SELECTION_AUTO_SELECT_LABEL "Auto-select latest file"

#define FILE_SELECTION_ANIMATION_LABEL "Animate"
#define FILE_SELECTION_ANIMATION_MIN_DURATION 100
#define FILE_SELECTION_ANIMATION_MAX_DURATION 1000


class FrameFileSelection : public FrameImGui {
public:
    FrameFileSelection(FrameGL * frameGL, std::shared_ptr<FrameInfo> frameInfo);
    ~FrameFileSelection();
    void render() override;
    std::vector<String> getBinFiles() const { return _binFiles; }
    String getCurrentFolderPath() const;
private:
    void loadFile(const String & filename);
    void scanBINFiles();
    void scanFolders();
    void scanDirectory(const String & path, std::vector<String> & results, bool filesOnly, const String & extension = "");
    void renderComboBox(const char * comboId, const std::vector<String> & items, int & selectedIndex, const char * noSelectionText, std::function<void(int)> onSelectionChanged = nullptr);
    size_t countRegions(const String & filepath);
    void selectPreviousFile();
    void selectNextFile();
    void startObserver();
    void stopObserver();
    void selectLatestFile();
    void startAnimation();
    void stopAnimation();
    void createAnimation();
private:
    std::shared_ptr<FrameInfo> _frameInfo;
    std::vector<String> _binFiles;
    std::vector<String> _folders;
    int _selectedFileIndex;
    int _selectedFolderIndex;
    RegionReader _regionReader;
    FrameGL * _frameGL;
    FileObserver _fileObserver;
    std::atomic<bool> _needsRescan;
    bool _autoSelectLatest;
    bool _animate;
    std::atomic<int> _animationDuration; 
    std::atomic<bool> _animationRunning;
    std::atomic<bool> _advanceRequested;
    std::thread _animationThread;
};

#endif

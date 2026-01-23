#include <frame/fileSelection/FrameFileSelection.hpp>
#include <frame/fileSelection/FileObserver.hpp>
#include <algorithm>
#include <chrono>


FrameFileSelection::FrameFileSelection(FrameGL * frameGL, std::shared_ptr<FrameInfo> frameInfo)
    : FrameImGui(FILE_SELECTION_DEFAULT_TITLE, true),
      _frameInfo(frameInfo),
      _selectedFileIndex(FILE_SELECTION_NO_INDEX),
      _selectedFolderIndex(FILE_SELECTION_DEFAULT_FOLDER_INDEX),
      _frameGL(frameGL),
      _needsRescan(false),
      _autoSelectLatest(false),
      _animate(false),
      _animationDuration(FILE_SELECTION_ANIMATION_MIN_DURATION),
      _animationRunning(false),
      _advanceRequested(false) {
    scanFolders();
    scanBINFiles();
    startObserver();
}

FrameFileSelection::~FrameFileSelection() {
    stopAnimation();
    stopObserver();
}

void FrameFileSelection::render() {
    if (!_isVisible) {
        return;
    }

    if (_needsRescan.exchange(false)) {
        scanBINFiles();
        if (_autoSelectLatest) {
            selectLatestFile();
        }
    }

    if (_advanceRequested.exchange(false)) {
        selectNextFile();
    }

    ImGuiIO & io = ImGui::GetIO();
    float scale = io.FontGlobalScale;
    ImVec2 windowSize(FILE_SELECTION_WINDOW_WIDTH * scale, FILE_SELECTION_WINDOW_HEIGHT * scale);
    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(windowSize, ImGuiCond_Always);
    
    ImGui::Begin(
        _title.c_str(), 
        &_isVisible,
        ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove 
    );

    ImGui::TextColored(IMGUI_TITLE_COLOR, FILE_SELECTION_TITLE);
    ImGui::Separator();
    
    float buttonSize = ImGui::GetFrameHeight();
    float spacing = ImGui::GetStyle().ItemSpacing.x;
    float buttonsWidth = 2 * buttonSize + spacing;
    
    ImGui::Text("%s", FILE_SELECTION_FOLDER_LABEL);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - buttonsWidth - spacing);
    renderComboBox(
        "##foldercombo",
        _folders,
        _selectedFolderIndex,
        FILE_SELECTION_NO_FOLDER,
        [this](int) {
            _selectedFileIndex = FILE_SELECTION_NO_INDEX;
            stopObserver();
            scanBINFiles();
            startObserver();
        }
    );
    
    ImGui::Spacing();
    
    ImGui::Text("%s", FILE_SELECTION_FILE_LABEL);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - buttonsWidth - spacing);
    renderComboBox(
        "##bincombo",
        _binFiles,
        _selectedFileIndex,
        FILE_SELECTION_NO_FILE,
        [this](int index) {
            loadFile(_binFiles[index]);
        }
    );
    
    ImGui::SameLine();
    ImGui::BeginDisabled(_binFiles.empty());
    if (ImGui::ArrowButton("##leftfile", ImGuiDir_Left)) {
        selectPreviousFile();
    }
    ImGui::SameLine();
    if (ImGui::ArrowButton("##rightfile", ImGuiDir_Right)) {
        selectNextFile();
    }
    ImGui::EndDisabled();

    ImGui::Spacing();
    ImGui::Checkbox(FILE_SELECTION_AUTO_SELECT_LABEL, &_autoSelectLatest);

    ImGui::Spacing();
    bool prevAnimate = _animate;
    if (ImGui::Checkbox(FILE_SELECTION_ANIMATION_LABEL, &_animate)) {
        if (_animate && !prevAnimate) {
            startAnimation();
        } else if (!_animate && prevAnimate) {
            stopAnimation();
        }
    }
    ImGui::SameLine();
    int duration = _animationDuration.load();
    if (ImGui::SliderInt("##animationDuration", &duration, FILE_SELECTION_ANIMATION_MIN_DURATION, FILE_SELECTION_ANIMATION_MAX_DURATION, "%d ms")) {
        _animationDuration.store(duration);
    }

    ImGui::End();
}

void FrameFileSelection::loadFile(const String& filename) {
    String filepath = getCurrentFolderPath() + filename;
    
    _regionReader.setRegionFilePath(filepath);
    Regions regions = _regionReader.read();
    size_t regionCount = regions.size();
    
    if (_frameInfo) {
        _frameInfo->setCurrentFile(filename);
        _frameInfo->setRegionCount(regionCount);
    }

    _frameGL->setRegions(regions);
}

void FrameFileSelection::scanBINFiles() {
    _binFiles.clear();
    scanDirectory(getCurrentFolderPath(), _binFiles, true, ".bin");
    std::sort(_binFiles.begin(), _binFiles.end(), naturalSort);
}

void FrameFileSelection::scanFolders() {
    _folders.clear();
    _folders.push_back(FILE_SELECTION_FOLDER_PREFIX);
    
    std::vector<String> subFolders;
    scanDirectory(FILE_SELECTION_RUNS_PATH, subFolders, false);
    
    for (const auto & folder : subFolders) {
        _folders.push_back(FILE_SELECTION_FOLDER_PREFIX + folder + "/");
    }
    
    std::sort(_folders.begin(), _folders.end());
}

void FrameFileSelection::scanDirectory(const String & path, std::vector<String> & results, bool filesOnly, const String & extension) {
    try {
        if (!std::filesystem::exists(path)) {
            return;
        }

        for (const std::filesystem::directory_entry & entry : std::filesystem::directory_iterator(path)) {
            if (filesOnly && entry.is_regular_file()) {
                String filename = entry.path().filename().string();
                if (extension.empty() || (filename.size() >= extension.size() && filename.substr(filename.size() - extension.size()) == extension)) {
                    results.push_back(filename);
                }
            } else if (!filesOnly && entry.is_directory()) {
                results.push_back(entry.path().filename().string());
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error scanning directory '" << path << "': " << e.what() << std::endl;
    }
}

void FrameFileSelection::renderComboBox(const char * comboId, const std::vector<String> & items, int & selectedIndex, const char * noSelectionText, std::function<void(int)> onSelectionChanged) {
    bool isSelected = (selectedIndex >= 0 && selectedIndex < static_cast<int>(items.size()));
    const char * currentSelection = isSelected ? items[selectedIndex].c_str() : noSelectionText;
    
    if (ImGui::BeginCombo(comboId, currentSelection)) {
        for (size_t i = 0; i < items.size(); i++) {
            const bool isItemSelected = (selectedIndex == static_cast<int>(i));
            if (ImGui::Selectable(items[i].c_str(), isItemSelected)) {
                selectedIndex = static_cast<int>(i);
                if (onSelectionChanged) {
                    onSelectionChanged(selectedIndex);
                }
            }

            if (isItemSelected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
}

String FrameFileSelection::getCurrentFolderPath() const {
    bool isValidFolder = (_selectedFolderIndex >= 0 && _selectedFolderIndex < static_cast<int>(_folders.size()));
    if (isValidFolder) {
        return "../" + _folders[_selectedFolderIndex];
    }
    return FILE_SELECTION_RUNS_PATH;
}

void FrameFileSelection::selectPreviousFile() {
    if (_binFiles.empty()) {
        return;
    }

    bool notFileSelected = _selectedFileIndex < 0;
    bool isFirstFileSelected = _selectedFileIndex == 0;
    if (notFileSelected || isFirstFileSelected) {
        _selectedFileIndex = static_cast<int>(_binFiles.size()) - 1;
    } else {
        _selectedFileIndex--;
    }

    bool isValidFileIndex = _selectedFileIndex >= 0 && _selectedFileIndex < static_cast<int>(_binFiles.size());
    if (isValidFileIndex) {
        loadFile(_binFiles[_selectedFileIndex]);
    }
}

void FrameFileSelection::selectNextFile() {
    if (_binFiles.empty()) {
        return;
    }

    bool notFileSelected = _selectedFileIndex < 0;
    bool isLastFileSelected = _selectedFileIndex >= static_cast<int>(_binFiles.size()) - 1;
    if (notFileSelected || isLastFileSelected) {
        _selectedFileIndex = 0;
    } else {
        _selectedFileIndex++;
    }

    bool isValidFileIndex = _selectedFileIndex >= 0 && _selectedFileIndex < static_cast<int>(_binFiles.size());
    if (isValidFileIndex) {
        loadFile(_binFiles[_selectedFileIndex]);
    }
}

void FrameFileSelection::startObserver() {
    _fileObserver.start(getCurrentFolderPath(), [this]() {
        _needsRescan = true;
    });
}

void FrameFileSelection::stopObserver() {
    _fileObserver.stop();
}

void FrameFileSelection::startAnimation() {
    if (_animationRunning.load()) return;
    _animationRunning.store(true);
    _animationThread = std::thread([this]() {
        while (_animationRunning.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(_animationDuration.load()));
            if (!_animationRunning.load()) break;
            _advanceRequested.store(true);
        }
    });
}

void FrameFileSelection::stopAnimation() {
    if (!_animationRunning.load()) return;
    _animationRunning.store(false);
    if (_animationThread.joinable()) {
        _animationThread.join();
    }
    _advanceRequested.store(false);
}

void FrameFileSelection::selectLatestFile() {
    if (_binFiles.empty()) {
        return;
    }

    _selectedFileIndex = static_cast<int>(_binFiles.size()) - 1;
    loadFile(_binFiles[_selectedFileIndex]);
}
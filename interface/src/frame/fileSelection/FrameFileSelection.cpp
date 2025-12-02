#include <frame/fileSelection/FrameFileSelection.hpp>
#include <frame/fileSelection/FileObserver.hpp>
#include <algorithm>


FrameFileSelection::FrameFileSelection(FrameGL * frameGL, std::shared_ptr<FrameInfo> frameInfo)
    : FrameImGui(FILE_SELECTION_DEFAULT_TITLE, true),
      _frameInfo(frameInfo),
      _selectedFileIndex(FILE_SELECTION_NO_INDEX),
      _selectedFolderIndex(FILE_SELECTION_DEFAULT_FOLDER_INDEX),
      _frameGL(frameGL),
      _needsRescan(false) {
    scanFolders();
    scanBINFiles();
    startObserver();
}

FrameFileSelection::~FrameFileSelection() {
    stopObserver();
}

void FrameFileSelection::render() {
    if (!_isVisible) {
        return;
    }

    if (_needsRescan.exchange(false)) {
        scanBINFiles();
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
        _csvFiles,
        _selectedFileIndex,
        FILE_SELECTION_NO_FILE,
        [this](int index) {
            loadFile(_csvFiles[index]);
        }
    );
    
    ImGui::SameLine();
    ImGui::BeginDisabled(_csvFiles.empty());
    if (ImGui::ArrowButton("##leftfile", ImGuiDir_Left)) {
        selectPreviousFile();
    }
    ImGui::SameLine();
    if (ImGui::ArrowButton("##rightfile", ImGuiDir_Right)) {
        selectNextFile();
    }
    ImGui::EndDisabled();

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
    _csvFiles.clear();
    scanDirectory(getCurrentFolderPath(), _csvFiles, true, ".bin");
    std::sort(_csvFiles.begin(), _csvFiles.end(), naturalSort);
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
    if (_csvFiles.empty()) {
        return;
    }

    bool notFileSelected = _selectedFileIndex < 0;
    bool isFirstFileSelected = _selectedFileIndex == 0;
    if (notFileSelected || isFirstFileSelected) {
        _selectedFileIndex = static_cast<int>(_csvFiles.size()) - 1;
    } else {
        _selectedFileIndex--;
    }

    bool isValidFileIndex = _selectedFileIndex >= 0 && _selectedFileIndex < static_cast<int>(_csvFiles.size());
    if (isValidFileIndex) {
        loadFile(_csvFiles[_selectedFileIndex]);
    }
}

void FrameFileSelection::selectNextFile() {
    if (_csvFiles.empty()) {
        return;
    }

    bool notFileSelected = _selectedFileIndex < 0;
    bool isLastFileSelected = _selectedFileIndex >= static_cast<int>(_csvFiles.size()) - 1;
    if (notFileSelected || isLastFileSelected) {
        _selectedFileIndex = 0;
    } else {
        _selectedFileIndex++;
    }

    bool isValidFileIndex = _selectedFileIndex >= 0 && _selectedFileIndex < static_cast<int>(_csvFiles.size());
    if (isValidFileIndex) {
        loadFile(_csvFiles[_selectedFileIndex]);
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
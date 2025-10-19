#include <frame/FrameFileSelection.hpp>
#include <algorithm>


FrameFileSelection::FrameFileSelection(FrameGL * frameGL, std::shared_ptr<FrameInfo> frameInfo, const String & title, bool visible)
    : ImGuiFrame(title, visible),
      _frameInfo(frameInfo),
      _selectedFileIndex(-1),
      _selectedFolderIndex(0),
      _frameGL(frameGL) {
    scanFolders();
    scanCSVFiles();
}

void FrameFileSelection::render() {
    if (!_isVisible) {
        return;
    }

    ImVec2 windowSize(300, 140);
    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(windowSize, ImGuiCond_FirstUseEver);
    
    ImGui::Begin(_title.c_str(), &_isVisible);

    ImGui::TextColored(ImVec4(0.2f, 0.8f, 1.0f, 1.0f), FRAME_FILE_SELECTION_TITLE);
    ImGui::Separator();
    
    renderComboBox(
        FRAME_FILE_SELECTION_FOLDER_LABEL,
        "##foldercombo",
        _folders,
        _selectedFolderIndex,
        FRAME_FILE_SELECTION_NO_FOLDER,
        [this](int) {
            _selectedFileIndex = -1;
            scanCSVFiles();
        }
    );
    
    ImGui::Spacing();
    
    renderComboBox(
        FRAME_FILE_SELECTION_LABEL,
        "##csvcombo",
        _csvFiles,
        _selectedFileIndex,
        FRAME_FILE_SELECTION_NO_FILE,
        [this](int index) {
            loadFile(_csvFiles[index]);
        }
    );

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

void FrameFileSelection::scanCSVFiles() {
    _csvFiles.clear();
    scanDirectory(getCurrentFolderPath(), _csvFiles, true, ".csv");
}

void FrameFileSelection::scanFolders() {
    _folders.clear();
    _folders.push_back("runs/");
    
    std::vector<String> subFolders;
    scanDirectory(FRAME_FILE_SELECTION_RUNS_PATH, subFolders, false);
    
    for (const auto & folder : subFolders) {
        _folders.push_back("runs/" + folder + "/");
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

        std::sort(results.begin(), results.end());
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error scanning directory '" << path << "': " << e.what() << std::endl;
    }
}

void FrameFileSelection::renderComboBox(const char * label, const char * comboId, const std::vector<String> & items, int & selectedIndex, const char * noSelectionText, std::function<void(int)> onSelectionChanged) {
    ImGui::Text("%s", label);
    
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
    return FRAME_FILE_SELECTION_RUNS_PATH;
}


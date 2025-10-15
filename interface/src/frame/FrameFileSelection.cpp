#include <frame/FrameFileSelection.hpp>
#include <algorithm>


FrameFileSelection::FrameFileSelection(FrameGL * frameGL, std::shared_ptr<FrameInfo> frameInfo, const String & title, bool visible)
    : ImGuiFrame(title, visible),
      _frameInfo(frameInfo),
      _selectedFileIndex(-1),
      _frameGL(frameGL) {
    scanCSVFiles();
}

void FrameFileSelection::render() {
    if (!_isVisible) {
        return;
    }

    ImGuiIO & io = ImGui::GetIO();
    ImVec2 windowSize(350, 120);
    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - windowSize.x, 0), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(windowSize, ImGuiCond_FirstUseEver);
    
    ImGui::Begin(_title.c_str(), &_isVisible);

    ImGui::TextColored(ImVec4(0.2f, 0.8f, 1.0f, 1.0f), FRAME_FILE_SELECTION_TITLE);
    ImGui::Separator();

    if (ImGui::Button(FRAME_FILE_SELECTION_REFRESH_BUTTON)) {
        scanCSVFiles();
    }

    ImGui::SameLine();
    
    const char * currentSelection = (_selectedFileIndex >= 0 && _selectedFileIndex < static_cast<int>(_csvFiles.size())) 
                                     ? _csvFiles[_selectedFileIndex].c_str() 
                                     : FRAME_FILE_SELECTION_NO_FILE;
    
    ImGui::Spacing();
    ImGui::Text(FRAME_FILE_SELECTION_LABEL);
    
    if (ImGui::BeginCombo("##csvcombo", currentSelection)) {
        for (size_t i = 0; i < _csvFiles.size(); i++) {
            const bool isSelected = (_selectedFileIndex == static_cast<int>(i));
            if (ImGui::Selectable(_csvFiles[i].c_str(), isSelected)) {
                _selectedFileIndex = static_cast<int>(i);
                loadFile(_csvFiles[i]);
            }

            if (isSelected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }

    ImGui::End();
}

void FrameFileSelection::loadFile(const String& filename) {
    String filepath = String(FRAME_FILE_SELECTION_RUNS_PATH) + filename;
    
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
    
    try {
        if (!std::filesystem::exists(FRAME_FILE_SELECTION_RUNS_PATH)) {
            return;
        }

        for (const std::filesystem::directory_entry & entry : std::filesystem::directory_iterator(FRAME_FILE_SELECTION_RUNS_PATH)) {
            if (entry.is_regular_file()) {
                String filename = entry.path().filename().string();
                if (filename.size() >= 4 && filename.substr(filename.size() - 4) == ".csv") {
                    _csvFiles.push_back(filename);
                }
            }
        }

        std::sort(_csvFiles.begin(), _csvFiles.end());
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error scanning CSV files: " << e.what() << std::endl;
    }
}


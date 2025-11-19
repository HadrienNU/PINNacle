#include <frame/fileSelection/FileObserver.hpp>
#include <thread>
#include <chrono>
#include <algorithm>
#include <iostream>

FileObserver::FileObserver()
    : _stopFlag(false), _isRunning(false) {
}

FileObserver::~FileObserver() {
    stop();
}

void FileObserver::start(const std::string& folderPath, Callback onFilesChangedCallback) {
    if (_isRunning) {
        return;
    }

    _folderPath = folderPath;
    _onFilesChangedCallback = onFilesChangedCallback;
    _stopFlag = false;
    _isRunning = true;
    
    {
        std::lock_guard<std::mutex> lock(_filesMutex);
        _trackedFiles.clear();
    }

    _watchThread = std::thread(&FileObserver::watchLoop, this);
}

void FileObserver::stop() {
    if (!_isRunning) {
        return;
    }

    _stopFlag = true;
    if (_watchThread.joinable()) {
        _watchThread.join();
    }
    _isRunning = false;
}

bool FileObserver::isRunning() const {
    return _isRunning;
}

void FileObserver::watchLoop() {
    while (!_stopFlag.load()) {
        std::filesystem::path folder = _folderPath;
        
        try {
            if (std::filesystem::exists(folder) && std::filesystem::is_directory(folder)) {
                std::set<std::filesystem::path> currentFiles;
                
                for (const auto& entry : std::filesystem::recursive_directory_iterator(
                    folder, 
                    std::filesystem::directory_options::skip_permission_denied
                )) {
                    if (entry.is_regular_file() && entry.path().extension() == ".csv") {
                        currentFiles.insert(entry.path());
                    }
                }

                std::set<std::filesystem::path> newFiles;
                {
                    std::lock_guard<std::mutex> lock(_filesMutex);
                    std::set_difference(
                        currentFiles.begin(), currentFiles.end(),
                        _trackedFiles.begin(), _trackedFiles.end(),
                        std::inserter(newFiles, newFiles.begin())
                    );
                    _trackedFiles = currentFiles;
                }

                if (!newFiles.empty() && _onFilesChangedCallback) {
                    _onFilesChangedCallback();
                }
            }
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Watcher error: " << e.what() << std::endl;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}


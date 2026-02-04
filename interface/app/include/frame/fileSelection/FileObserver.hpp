#pragma once
#include <string>
#include <functional>
#include <atomic>
#include <thread>
#include <set>
#include <filesystem>
#include <mutex>

class FileObserver {
public:
    using Callback = std::function<void()>;
    FileObserver();
    ~FileObserver();
    void start(const std::string& folderPath, Callback onFilesChangedCallback);
    void stop();
    bool isRunning() const;
private:
    void watchLoop();
private:
    std::string _folderPath;
    Callback _onFilesChangedCallback;
    std::thread _watchThread;
    std::atomic<bool> _stopFlag;
    std::atomic<bool> _isRunning;
    std::set<std::filesystem::path> _trackedFiles;
    std::set<std::filesystem::path> _trackedDirectories;
    std::mutex _filesMutex;
};

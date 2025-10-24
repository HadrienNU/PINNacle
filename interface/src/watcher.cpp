#include <iostream>
#include <watcher.hpp>
#include <thread>
#include <chrono>
#include <filesystem>
#include <set>
#include <algorithm>

void runWatcher(const std::string & folder_path){
    std::set<std::filesystem::path> s{};
    while (true){
        std::filesystem::path folder = folder_path;
        if (std::filesystem::is_directory(folder)){
            std::set<std::filesystem::path> current;
            for (const auto & entry : std::filesystem::directory_iterator(folder)){
                std::filesystem::path filepath = entry.path();
                if (std::filesystem::is_regular_file(filepath)){
                    if (filepath.extension() == ".csv"){
                        current.insert(filepath);
                    }
                }
            }
            std::set<std::filesystem::path> newFiles;
            std::set_difference(current.begin(), current.end(),
                                s.begin(), s.end(),
                                std::inserter(newFiles, newFiles.begin()));
            for (const auto& f : newFiles){
                std::cout << "[NEW FILE]: " << f.string() << std::endl;
            }
            s = current;
        }
        else{
            continue;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

    }
}


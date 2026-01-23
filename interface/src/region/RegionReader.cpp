#include <region/RegionReader.hpp>
#include <iostream>

RegionReader::RegionReader() {
    _regionFilePath = "";
}

void RegionReader::setRegionFilePath(const String & regionFilePath) {
    _regionFilePath = regionFilePath;
}

Regions RegionReader::read() const {
    Regions regions;

    if (_regionFilePath.empty()) return regions;

    std::ifstream regionFile(_regionFilePath, std::ios::binary);
    if (!regionFile.is_open()) return regions;

    // ---- LECTURE DU HEADER ----
    Header header;
    regionFile.read(reinterpret_cast<char*>(&header), sizeof(Header));

    int dim = header.dim;

    // ---- LECTURE DES BODIES ----
    Body body;

    Region currentRegion(-1);
    int currentIdRegion = -1;
    
    const int STATS_MARKER = -999999999;

    while (regionFile.read(reinterpret_cast<char*>(&body), sizeof(Body))) {
        int idRegion = body.id;
        
        if (idRegion == STATS_MARKER) {
            break;
        }

        if (currentIdRegion == -1) {
            currentIdRegion = idRegion;
            currentRegion = Region(currentIdRegion, dim);
        }

        if (idRegion != currentIdRegion) {
            regions.push_back(currentRegion);

            currentIdRegion = idRegion;
            currentRegion = Region(currentIdRegion, dim);
        }

        glm::vec3 point(body.x, body.y, body.z);
        currentRegion.addPoint(point);
    }

    if (currentIdRegion != -1 && !currentRegion.getPoints().empty()) {
        regions.push_back(currentRegion);
    }
    
    bool hasStats = !regionFile.eof();
    
    if (hasStats) {
        regionFile.clear();
        
        std::map<int, std::map<std::string, float>> allStats;
        readStatistics(regionFile, allStats);
    
        for (auto& region : regions) {
            int regionId = region.getId();
            if (allStats.find(regionId) != allStats.end()) {
                for (const auto& stat : allStats[regionId]) {
                    region.setStatistic(stat.first, stat.second);
                }
            }
        }
    }

    regionFile.close();
    return regions;
}

void RegionReader::readStatistics(std::ifstream& file, std::map<int, std::map<std::string, float>>& allStats) const {

    int numStatTypes = 0;
    file.read(reinterpret_cast<char*>(&numStatTypes), sizeof(int));
    
    if (file.fail() || numStatTypes <= 0) {
        return; 
    }
    
    for (int i = 0; i < numStatTypes; i++) {
        int nameLength = 0;
        file.read(reinterpret_cast<char*>(&nameLength), sizeof(int));
        
        if (file.fail() || nameLength <= 0) {
            return;
        }
        
        std::string statName(nameLength, '\0');
        file.read(&statName[0], nameLength);
        
        if (file.fail()) {
            return;
        }
        
        int numRegions = 0;
        file.read(reinterpret_cast<char*>(&numRegions), sizeof(int));
        
        if (file.fail() || numRegions < 0) {
            return;
        }
        
        for (int j = 0; j < numRegions; j++) {
            int regionId = 0;
            float value = 0.0f;
            file.read(reinterpret_cast<char*>(&regionId), sizeof(int));
            file.read(reinterpret_cast<char*>(&value), sizeof(float));
            
            if (file.fail()) {
                return;
            }
            
            allStats[regionId][statName] = value;
        }
    }
}

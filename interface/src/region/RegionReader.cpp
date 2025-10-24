#include <region/RegionReader.hpp>

RegionReader::RegionReader() {
    _regionFilePath = "";
}

void RegionReader::setRegionFilePath(const String & regionFilePath) {
    _regionFilePath = regionFilePath;
}

Regions RegionReader::read() const {
    Regions regions;
    if (_regionFilePath.empty()) {
        return regions;
    }
    std::ifstream regionFile(_regionFilePath);
    if (!regionFile.is_open()) {
        return regions;
    }

    String line;
    Region currentRegion(-1); // This region does not exist
    int currentIdRegion = -1;
    bool header = true;
    while (std::getline(regionFile, line)) {
        if (header) {
            header = false;
            continue;
        }
        std::stringstream ss(line);
        String x, y, z, idRegion;

        std::getline(ss, x, ',');
        std::getline(ss, y, ',');
        std::getline(ss, z, ',');
        std::getline(ss, idRegion, ',');

        glm::vec3 point(std::stof(x), std::stof(y), std::stof(z));
        currentRegion.addPoint(point);

        /* Changing region */
        if (currentIdRegion == -1) {
            currentIdRegion = std::stoi(idRegion);
            currentRegion = Region(currentIdRegion);
        }
        if (std::stoi(idRegion) != currentIdRegion) {
            currentIdRegion = std::stoi(idRegion);
            regions.push_back(currentRegion);
            currentRegion = Region(currentIdRegion);
        }
    }

    regionFile.close();
    return regions;
}

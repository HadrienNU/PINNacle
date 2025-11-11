#include <region/RegionReader.hpp>

RegionReader::RegionReader() {
    _regionFilePath = "";
}

void RegionReader::setRegionFilePath(const String & regionFilePath) {
    _regionFilePath = regionFilePath;
}

Regions RegionReader::read() const {
    Regions regions;
    if (_regionFilePath.empty()) return regions;

    std::ifstream regionFile(_regionFilePath);
    if (!regionFile.is_open()) return regions;

    String line;
    bool header = true;

    Region currentRegion(-1);
    int currentIdRegion = -1;

    while (std::getline(regionFile, line)) {
        if (header) { header = false; continue; }

        std::stringstream ss(line);
        String x, y, z, idRegionStr;

        std::getline(ss, x, ',');
        std::getline(ss, y, ',');
        std::getline(ss, z, ',');
        std::getline(ss, idRegionStr, ',');

        int idRegion = std::stoi(idRegionStr);

        if (currentIdRegion == -1) {
            currentIdRegion = idRegion;
            currentRegion = Region(currentIdRegion);
        } else if (idRegion != currentIdRegion) {
            regions.push_back(currentRegion);
            currentIdRegion = idRegion;
            currentRegion = Region(currentIdRegion);
        }

        glm::vec3 point(std::stof(x), std::stof(y), std::stof(z));
        currentRegion.addPoint(point);
    }

    if (currentIdRegion != -1 && !currentRegion.getPoints().empty()) {
        regions.push_back(currentRegion);
    }

    regionFile.close();
    return regions;
}

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

    while (regionFile.read(reinterpret_cast<char*>(&body), sizeof(Body))) {

        int idRegion = body.id;

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

    regionFile.close();
    return regions;
}

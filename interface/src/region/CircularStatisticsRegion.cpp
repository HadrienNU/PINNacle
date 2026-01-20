#include "CircularStatisticsRegion.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

bool CircularStatisticsRegion::appendAreasToBinary(const std::string& filename, float domainRadius) {
    
    std::fstream file(filename, std::ios::in | std::ios::out | std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Erreur ouverture fichier : " << filename << std::endl;
        return false;
    }

    int dim;
    if (!file.read(reinterpret_cast<char*>(&dim), sizeof(int))) return false;

    std::vector<PointRecord> points;
    PointRecord tempPoint;
    
    while (file.read(reinterpret_cast<char*>(&tempPoint), sizeof(PointRecord))) {
        points.push_back(tempPoint);
    }
    file.clear();

    if (points.empty()) {
        std::cerr << "Aucun point lu." << std::endl;
        return false;
    }

    std::vector<float> regionAreas;
    std::vector<PointRecord> currentRegion;
    
    if (!points.empty()) {
        int currentId = points[0].id;
        auto processRegion = [&](const std::vector<PointRecord>& pts) {
            float area = 0.0f;
            if (isRegionInsideCircle(pts, domainRadius)) {
                for (size_t i = 0; i < pts.size(); i += 3) {
                    if (i + 2 < pts.size()) {
                        area += calculateTriangleArea(
                            pts[i].x, pts[i].y, pts[i+1].x, pts[i+1].y, pts[i+2].x, pts[i+2].y
                        );
                    }
                }
            } else {
                area = -1.0f;
            }
            regionAreas.push_back(area);
        };

        for (const auto& p : points) {
            if (p.id != currentId) {
                processRegion(currentRegion);
                currentRegion.clear();
                currentId = p.id;
            }
            currentRegion.push_back(p);
        }
        processRegion(currentRegion);
    }

    long dataEndPos = sizeof(int) + (points.size() * sizeof(PointRecord));
    file.seekp(dataEndPos, std::ios::beg);
    file.write(reinterpret_cast<const char*>(regionAreas.data()), regionAreas.size() * sizeof(float));
    int numRegions = static_cast<int>(regionAreas.size());
    file.write(reinterpret_cast<const char*>(&numRegions), sizeof(int));
    long finalSize = file.tellp();
    file.close();

    std::cout << "[CircularStatisticsRegion] Succes : " << numRegions << " aires ajoutees." << std::endl; 
    return true;
}


float CircularStatisticsRegion::calculateTriangleArea(float x1, float y1, float x2, float y2, float x3, float y3) {
    return 0.5f * std::abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2));
}

bool CircularStatisticsRegion::isRegionInsideCircle(const std::vector<PointRecord>& pts, float radius) {
    if (pts.empty()) return false;
    
    double sumX = 0;
    double sumY = 0;
    
    for (const auto& p : pts) {
        sumX += p.x;
        sumY += p.y;
    }
    
    float cx = static_cast<float>(sumX / pts.size());
    float cy = static_cast<float>(sumY / pts.size());
    
    return (cx * cx + cy * cy) <= (radius * radius);
}
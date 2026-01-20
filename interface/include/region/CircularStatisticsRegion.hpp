#ifndef CIRCULAR_STATISTICS_REGION_HPP
#define CIRCULAR_STATISTICS_REGION_HPP

#include <string>
#include <vector>

class CircularStatisticsRegion {
public:
    static bool appendAreasToBinary(const std::string& filename, float domainRadius = 1.0f);

private:
    struct PointRecord {
        float x, y, z;
        int id;
    };
    
    static float calculateTriangleArea(float x1, float y1, float x2, float y2, float x3, float y3);
    static bool isRegionInsideCircle(const std::vector<PointRecord>& pts, float radius);
};

#endif
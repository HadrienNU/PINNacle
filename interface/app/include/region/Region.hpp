#ifndef __REGION_HPP__
#define __REGION_HPP__


#include <glm/glm.hpp>
#include <vector>
#include <map>
#include <string>


class Region {
public:
    Region(int id, int dim = 2);
    void addPoint(const glm::vec2 & point);
    void addPoint(const glm::vec3 & point);
    std::vector<glm::vec3> getPoints() { return _points; }
    std::vector<glm::vec3> createMesh() const;
    int getId() const { return _id; }
    void setStatistic(const std::string& name, float value);
    std::map<std::string, float> getStatistics() const { return _statistics; }
private:
    int _id;
    int _dim;
    std::vector<glm::vec3> _points;
    std::map<std::string, float> _statistics;
};

typedef std::vector<Region> Regions;

#endif

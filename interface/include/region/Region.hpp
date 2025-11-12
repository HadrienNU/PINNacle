#ifndef __REGION_HPP__
#define __REGION_HPP__


#include <glm/glm.hpp>
#include <vector>


class Region {
public:
    Region(int id);
    void addPoint(const glm::vec2 & point);
    void addPoint(const glm::vec3 & point);
    std::vector<glm::vec3> getPoints() { return _points; }
    std::vector<glm::vec3> createMesh() const;
    int getId() const { return _id; }
private:
    int _id;
    std::vector<glm::vec3> _points;
};

typedef std::vector<Region> Regions;

#endif

#ifndef __REGION_HPP__
#define __REGION_HPP__


#include <glm/glm.hpp>
#include <vector>


class Region {
public:
    Region();
    void addPoint(const glm::vec2 & point);
    void addPoint(const glm::vec3 & point);
    glm::vec3 getCenter() const;
private:
    std::vector<glm::vec3> _points;
};

#endif

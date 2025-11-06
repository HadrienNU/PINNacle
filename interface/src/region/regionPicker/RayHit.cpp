#include <region/regionPicker/RayHit.hpp>


RayHit::RayHit() 
    : hit(false), 
      t(std::numeric_limits<float>::max()), 
      regionId(-1) {}

bool RayHit::isCloser(const RayHit& other) const {
    return hit && t < other.t;
}

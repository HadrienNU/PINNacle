#ifndef __RAY_HIT_HPP__
#define __RAY_HIT_HPP__


#include <limits>


struct RayHit {
    bool hit;
    float t;
    int regionId;
    RayHit();
    bool isCloser(const RayHit& other) const;
};

#endif

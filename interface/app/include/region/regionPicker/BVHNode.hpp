#ifndef __BVH_NODE_HPP__
#define __BVH_NODE_HPP__


#include <region/regionPicker/AABB.hpp>


struct BVHNode {
    AABB bounds;
    int leftChild;
    int rightChild;
    int firstTriangle;
    int triangleCount;
    BVHNode();
    bool isLeaf() const;
};

#endif

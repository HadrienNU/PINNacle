#include <region/regionPicker/BVHNode.hpp>


BVHNode::BVHNode() 
    : leftChild(-1), 
      rightChild(-1), 
      firstTriangle(0), 
      triangleCount(0) {}

bool BVHNode::isLeaf() const {
    return triangleCount > 0;
}

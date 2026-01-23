#ifndef __REGION_READER_HPP__
#define __REGION_READER_HPP__

#include <util.hpp>
#include <region/Region.hpp>
#include <map>
#include <string>


struct Header {
    int dim;
};

struct Body {
    float x;
    float y;
    float z;
    int id;
};

class RegionReader {
public:
    RegionReader();
    void setRegionFilePath(const String & regionFilePath);
    Regions read() const;
private:
    String _regionFilePath;
    void readStatistics(std::ifstream& file, std::map<int, std::map<std::string, float>>& allStats) const;
};

#endif

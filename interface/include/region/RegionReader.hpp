#ifndef __REGION_READER_HPP__
#define __REGION_READER_HPP__

#include <util.hpp>
#include <region/Region.hpp>


class RegionReader {
public:
    RegionReader();
    void setRegionFilePath(const String & regionFilePath);
    Regions read() const;
private:
    String _regionFilePath;
};

#endif

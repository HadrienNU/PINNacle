#include <frame/MainFrame.hpp>
#include <region/RegionReader.hpp>
#include <ctime>  


int main() {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));    
    
    RegionReader regionReader;
    regionReader.setRegionFilePath("../runs/epoch100.csv");
    Regions regions = regionReader.read();

    FrameGL frameGL = FrameGL(Color(30, 30, 30));
    MainFrame mainFrame("PINNacle Interface", {800, 800}, frameGL);
    frameGL.setRegions(regions);
    
    mainFrame.run();
    return EXIT_SUCCESS;
}

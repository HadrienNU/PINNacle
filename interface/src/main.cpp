#include <frame/MainFrame.hpp>
#include <region/RegionReader.hpp>
#include <ctime>  


int main() {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    MainFrame mainFrame("PINNacle Interface", {800, 800}, Color(30, 30, 30));
    
    RegionReader regionReader;
    regionReader.setRegionFilePath("../runs/epoch100.csv");
    Regions regions = regionReader.read();
    
    mainFrame.run();
    return EXIT_SUCCESS;
}

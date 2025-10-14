#include <frame/MainFrame.hpp>
#include <region/RegionReader.hpp>
#include <ctime>  


int main() {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));    
    
    RegionReader regionReader;
    regionReader.setRegionFilePath("../runs/epoch100.csv");
    Regions regions = regionReader.read();

    FrameGL * frameGL = new FrameGL(Color(30, 30, 30));
    ImGuiFrames imguiFrames;

    MainFrame mainFrame("PINNacle Interface", {800, 800}, frameGL, imguiFrames);
    frameGL -> setRegions(regions);
    
    mainFrame.run();
    delete frameGL;
    return EXIT_SUCCESS;
}

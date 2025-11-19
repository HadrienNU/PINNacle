#include <frame/MainFrame.hpp>
#include <ctime>  
#include <thread>


#define BACKGROUND_COLOR Color(30, 30, 30)
#define WINDOW_TITLE "PINNacle Interface"
#define WINDOW_WIDTH 1600
#define WINDOW_HEIGHT 900


int main() {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));    

    FrameGL * frameGL = new FrameGL(BACKGROUND_COLOR);
    FramesImGui imguiFrames(frameGL);
    MainFrame mainFrame(WINDOW_TITLE, {WINDOW_WIDTH, WINDOW_HEIGHT}, frameGL, imguiFrames);
    mainFrame.run();
    delete frameGL;
    return EXIT_SUCCESS;
}

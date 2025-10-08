#include <frame/MainFrame.hpp>
#include <ctime>  


int main() {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    MainFrame mainFrame("PINNacle Interface", {800, 800}, generateRandomColor());
    mainFrame.run();
    return EXIT_SUCCESS;
}

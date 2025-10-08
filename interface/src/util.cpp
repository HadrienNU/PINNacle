#include <util.hpp>

void error(const String & errorMessage) {
    std::cerr << errorMessage << std::endl;
    throw std::runtime_error(errorMessage);
}

Color::Color(float red, float green, float blue) {
    r = red;
    g = green;
    b = blue;
}

Color generateRandomColor() {
    float red   = static_cast<float>(std::rand()) / RAND_MAX;
    float green = static_cast<float>(std::rand()) / RAND_MAX;
    float blue  = static_cast<float>(std::rand()) / RAND_MAX;
    return Color(red, green, blue);
}

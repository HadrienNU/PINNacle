#include "color.hpp"

Color::Color() {
    r = 1.0f;
    g = 1.0f;
    b = 1.0f;
}


Color::Color(float red, float green, float blue) {
    r = red;
    g = green;
    b = blue;
}

Color Color::generateRandomColor() {
    static bool initialized = false;
    if (!initialized) {
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
        initialized = true;
    }

    float red   = static_cast<float>(std::rand()) / RAND_MAX;
    float green = static_cast<float>(std::rand()) / RAND_MAX;
    float blue  = static_cast<float>(std::rand()) / RAND_MAX;

    return Color(red, green, blue);
}



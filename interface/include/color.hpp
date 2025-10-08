#ifndef COLOR_HPP
#define COLOR_HPP

#include <cstdlib>
#include <ctime>  

struct Color {
    float r, g, b;
    Color();
    Color(float red, float green, float blue);
    static Color generateRandomColor();
};

#endif

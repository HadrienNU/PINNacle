#ifndef __UTIL_HPP__
#define __UTIL_HPP__


#include <iostream>


struct Size {
    int width;
    int height;
};

struct Color {
    float r, g, b;
    Color(float red, float green, float blue);
};

typedef std::string String;

void error(const String & errorMessage);

Color generateRandomColor();

#endif


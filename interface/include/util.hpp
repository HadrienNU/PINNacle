#ifndef __UTIL_HPP__
#define __UTIL_HPP__


#include <iostream>
#include <GL/gl.h>


struct Size {
    int width;
    int height;
};

struct Color {
    unsigned char r, g, b;
    Color(unsigned char red, unsigned char green, unsigned char blue);
};

struct GLColor {
    GLfloat r, g, b, a;
    GLColor(const Color& color, GLfloat alpha = 1.0f);
};

typedef std::string String;

void error(const String & errorMessage);

Color generateRandomColor();

#endif


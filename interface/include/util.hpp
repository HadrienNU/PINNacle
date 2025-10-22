#ifndef __UTIL_HPP__
#define __UTIL_HPP__


#include <glad/glad.h>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <glm/glm.hpp>


struct Size {
    int width;
    int height;
};

struct Color {
    unsigned char r = 0;
    unsigned char g = 0;
    unsigned char b = 0;
    Color() {}
    Color(unsigned char red, unsigned char green, unsigned char blue);
};

struct ColorGL {
    GLfloat r, g, b, a;
    ColorGL(const Color& color, GLfloat alpha = 1.0f);
};

typedef std::unordered_map<int, Color> TableColor;
typedef std::string String;

void error(const String & errorMessage);

Color generateRandomColor();

#endif


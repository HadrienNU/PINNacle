#ifndef __UTIL_HPP__
#define __UTIL_HPP__


#include <glad/glad.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <regex>
#include <glm/glm.hpp>


struct Size {
    int width;
    int height;
};

struct Color {
    unsigned char r, g, b;
    Color(unsigned char red, unsigned char green, unsigned char blue);
};

struct ColorGL {
    GLfloat r, g, b, a;
    ColorGL(const Color& color, GLfloat alpha = 1.0f);
};

typedef std::vector<Color> TableColor;
typedef std::string String;

void error(const String & errorMessage);

Color generateRandomColor();

TableColor generateTableColor(size_t tableColorSize);
void resizeTableColor(TableColor & tableColor, size_t size);

bool naturalSort(const String & a, const String & b);

#endif


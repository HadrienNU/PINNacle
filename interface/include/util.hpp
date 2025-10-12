#ifndef __UTIL_HPP__
#define __UTIL_HPP__


#if defined(_WIN32)
#  define NOMINMAX
#  include <windows.h>
#  include <GL/gl.h>
#elif defined(__APPLE__)
# ifndef GL_SILENCE_DEPRECATED
#   define GL_SILENCE_DEPRECATED 
#  endif
#  include <OpenGL/gl3.h>
#else
#  include <GL/gl.h>
#endif

#include <vector>
#include <iostream>


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

typedef std::vector<Color> TableColor;
typedef std::string String;

void error(const String & errorMessage);

Color generateRandomColor();

#endif


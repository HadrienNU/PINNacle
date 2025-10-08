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


#include <util.hpp>

void error(const String & errorMessage) {
    std::cerr << errorMessage << std::endl;
    throw std::runtime_error(errorMessage);
}

Color::Color(unsigned char red, unsigned char green, unsigned char blue) {
    r = red;
    g = green;
    b = blue;
}

Color generateRandomColor() {
    unsigned char red   = static_cast<unsigned char>(std::rand() % 256);
    unsigned char green = static_cast<unsigned char>(std::rand() % 256);
    unsigned char blue  = static_cast<unsigned char>(std::rand() % 256);
    return Color(red, green, blue);
}

ColorGL::ColorGL(const Color& color, GLfloat alpha) {
    r = static_cast<GLfloat>(color.r) / 255.0f;
    g = static_cast<GLfloat>(color.g) / 255.0f;
    b = static_cast<GLfloat>(color.b) / 255.0f;
    a = alpha;
}

TableColor generateTableColor(size_t tableColorSize) {
    TableColor tableColor;
    for (size_t i = 0; i < tableColorSize; i++) {
        Color randomColor = generateRandomColor();
        tableColor.push_back(randomColor);
    }
    return tableColor;
}

void resize(TableColor& tableColor, size_t size) {
    if (tableColor.size() < size) {
        size_t missingColors = size - tableColor.size();
        for (size_t i = 0; i < missingColors; ++i) {
            tableColor.push_back(generateRandomColor());
        }
    }
}


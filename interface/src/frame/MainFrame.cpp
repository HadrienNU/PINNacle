#include <frame/MainFrame.hpp>


static void error_callback(int error, const char * description) {
    std::cerr << "GLFW Error (" << error << "): " << description << std::endl;
}

static void framebuffer_size_callback(GLFWwindow * window, int width, int height) {
    MainFrame * frame = static_cast<MainFrame*>(glfwGetWindowUserPointer(window));
    if (!frame) {
        return;
    }
    frame -> resize({width, height});
}

static void scroll_callback(GLFWwindow *window, double /*xoffset*/, double yoffset) {
    MainFrame *frame = static_cast<MainFrame *>(glfwGetWindowUserPointer(window));
    if (!frame || !frame->getFrameGL()) {
        return;
    }
    frame->getFrameGL()->zoom(static_cast<float>(yoffset));
}

static bool rightButtonPressed = false;
static double lastX = 0.0, lastY = 0.0;
static bool leftButtonPressed = false;

static void mouse_button_callback(GLFWwindow* window, int button, int action, int /*mods*/) {
    MainFrame* frame = static_cast<MainFrame*>(glfwGetWindowUserPointer(window));
    if (!frame) return;

    if (button == GLFW_MOUSE_BUTTON_RIGHT)
        rightButtonPressed = (action == GLFW_PRESS);

    if (button == GLFW_MOUSE_BUTTON_LEFT)
        leftButtonPressed = (action == GLFW_PRESS);

    if (action == GLFW_PRESS)
        glfwGetCursorPos(window, &lastX, &lastY);
}


static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    MainFrame* frame = static_cast<MainFrame*>(glfwGetWindowUserPointer(window));
    if (!frame) return;

    static const float ROTATION_SPEED = 0.005f;

    double dx = xpos - lastX;
    double dy = ypos - lastY;
    lastX = xpos;
    lastY = ypos;

    if (leftButtonPressed) {
        frame->getFrameGL()->rotateCamera(dx * ROTATION_SPEED, dy * ROTATION_SPEED);
    } else if (rightButtonPressed) {
        frame->getFrameGL()->translateCamera(dx * 0.01f, -dy * 0.01f);
    }
}



MainFrame::MainFrame(const String & title, const Size & frameSize, FrameGL * frameGL, ImGuiFrames & imguiFrames) : 
_title(title), _frameSize(frameSize), _frameGL(frameGL), _imguiFrames(imguiFrames) {
    init();
}

MainFrame::~MainFrame() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(_window);
    glfwTerminate();    
}

void MainFrame::resize(const Size & frameSize) {
    _frameSize = frameSize;
    _frameGL -> resize(_frameSize);
    glViewport(0, 0, frameSize.width, frameSize.height);
}

void MainFrame::init() {
    glfwSetErrorCallback(error_callback);

    if (!glfwInit()) {
        error("Failed to initialize GLFW");
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 16);  
#if __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    _window = glfwCreateWindow(
        _frameSize.width, _frameSize.height, _title.c_str(), 
        nullptr, nullptr
    );
    if (!_window) {
        glfwTerminate();
        error("Failed to create GLFW window");        
    }
    
    glfwMakeContextCurrent(_window);
    gladLoadGL();
    glfwSwapInterval(1);

    int fbW, fbH;
    glfwSetWindowUserPointer(_window, this);
    glfwGetFramebufferSize(_window, &fbW, &fbH);
    glViewport(0, 0, fbW, fbH);
    glfwSetFramebufferSizeCallback(_window, framebuffer_size_callback);
    glfwSetScrollCallback(_window, scroll_callback);
    glfwSetMouseButtonCallback(_window, mouse_button_callback);
    glfwSetCursorPosCallback(_window, cursor_position_callback);
    initImGUI();  
    _frameGL -> init(_frameSize);
}

void MainFrame::initImGUI() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO & io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(_window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
}

void MainFrame::run() {
    while (!glfwWindowShouldClose(_window)) {
        glfwPollEvents();


        _imguiFrames.render();
        _frameGL -> render();
        
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        
        glfwSwapBuffers(_window);
    }
}

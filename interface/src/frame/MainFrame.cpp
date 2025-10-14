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
    glfwSwapInterval(1); // vsync

    int fbW, fbH;
    glfwSetWindowUserPointer(_window, this);
    glfwGetFramebufferSize(_window, &fbW, &fbH);
    glViewport(0, 0, fbW, fbH);
    glfwSetFramebufferSizeCallback(_window, framebuffer_size_callback);
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

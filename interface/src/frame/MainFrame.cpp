#include <frame/MainFrame.hpp>


static void error_callback(int error, const char * description) {
    std::cerr << "GLFW Error (" << error << "): " << description << std::endl;
}

static void framebuffer_size_callback(GLFWwindow * window, int width, int height) {
    (void) window;
    glViewport(0, 0, width, height);
}


MainFrame::MainFrame(const String & title, const Size & frameSize) : 
_title(title), _frameSize(frameSize) {
    init();
}

MainFrame::~MainFrame() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(_window);
    glfwTerminate();
}

void MainFrame::init() {
    glfwSetErrorCallback(error_callback);

    if (!glfwInit()) {
        error("Failed to initialize GLFW");
    }

    // Request an OpenGL 3.3 Core profile context
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
    glfwSwapInterval(1); // vsync

    int fbW, fbH;
    glfwGetFramebufferSize(_window, &fbW, &fbH);
    glViewport(0, 0, fbW, fbH);
    glfwSetFramebufferSizeCallback(_window, framebuffer_size_callback);

    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    //IMGUI_CHECKVERSION();
    //ImGui::CreateContext();
    //ImGuiIO & io = ImGui::GetIO(); (void)io;
    //ImGui::StyleColorsDark();
    //ImGui_ImplGlfw_InitForOpenGL(_window, true);
    //ImGui_ImplOpenGL3_Init("#version 330");
}

void MainFrame::runImGui() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("Test Window");
    ImGui::Text("This is some useful text.");
    static float value = 0.0f;
    ImGui::SliderFloat("float", &value, 0.0f, 1.0f);
    if (ImGui::Button("Close"))
        glfwSetWindowShouldClose(_window, GLFW_TRUE);
    ImGui::End();

    ImGui::Render();
}


void MainFrame::runOpenGL() {
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
}

void MainFrame::run() {
    while (!glfwWindowShouldClose(_window)) {
        glfwPollEvents();

        //runImGui();
        runOpenGL();

        //ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        
        glfwSwapBuffers(_window);
    }
}

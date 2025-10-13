#include <frame/MainFrame.hpp>


static void error_callback(int error, const char * description) {
    std::cerr << "GLFW Error (" << error << "): " << description << std::endl;
}

static void framebuffer_size_callback(GLFWwindow * window, int width, int height) {
    (void) window;
    glViewport(0, 0, width, height);
}

MainFrame::MainFrame(const String & title, const Size & frameSize, const Color & backgroundColor, Regions regions) : 
_title(title), _frameSize(frameSize), _backgroundColor(backgroundColor), _regions(regions) {
    init();
}

MainFrame::~MainFrame() {
    delete _shader;
    delete[] _vaos;
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
    glfwGetFramebufferSize(_window, &fbW, &fbH);
    glViewport(0, 0, fbW, fbH);
    glfwSetFramebufferSizeCallback(_window, framebuffer_size_callback);
    initImGUI();  
    
    _shader = new Shader();
    _vaos = new VAO[_regions.size()];
    _tableColor = generateTableColor(_regions.size());

    for (size_t i = 0; i < _regions.size(); i ++) {
        _vaos[i].setVector(0, _regions[i].createMesh());
    }
}

void MainFrame::initImGUI() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO & io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(_window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
}

void MainFrame::runImGui() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("Test Window");
    ImGui::Text("This is some useful text.");
    static float value = 0.0f;
    ImGui::SliderFloat("float", &value, 0.0f, 1.0f);
    if (ImGui::Button("Close")) {
        glfwSetWindowShouldClose(_window, GLFW_TRUE);
    }
    ImGui::End();

    ImGui::Render();
}

void MainFrame::runOpenGL() {
    ColorGL colorGL = ColorGL(_backgroundColor);
    glClearColor(colorGL.r, colorGL.g, colorGL.b, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    _shader -> bind();
    for (size_t i = 0; i < _regions.size(); i ++) {
        ColorGL colorGL = ColorGL(_tableColor[i]);
        glm::vec3 color(colorGL.r, colorGL.g, colorGL.b);
        _shader -> setUniformVector("color", color);
        _vaos[i].bind();
        glDrawArrays(GL_TRIANGLES, 0, 6);
    }        
}

void MainFrame::run() {
    while (!glfwWindowShouldClose(_window)) {
        glfwPollEvents();

        runImGui();
        runOpenGL();

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        
        glfwSwapBuffers(_window);
    }
}

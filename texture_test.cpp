#include <GL/glew.h>
#include <GL/glut.h>

#include <cv.h>
#include <highgui.h>

#include <iostream>

GLuint g_texture; //the array for our texture

cv::Mat make_rgba(const cv::Mat& image, cv::Mat alpha = cv::Mat()) {
  if (alpha.empty()) {
    alpha = cv::Mat(cv::Mat::ones(image.size(), CV_8U) * 255);
  }
  static int fromto[] = {0, 2,  1, 1,  2, 0,  3, 3};
  cv::Mat with_alpha(image.size(), CV_8UC4);
  cv::Mat images[] = {image, alpha};
  cv::mixChannels(images, 2, &with_alpha, 1, fromto, 4);
  return with_alpha;
}

GLuint LoadTexture( cv::Mat image )
{
    GLuint texture;

    //The following code will read in our RAW file

    glGenTextures(1, &texture); //generate the texture with the loaded data
    glBindTexture(GL_TEXTURE_2D, texture); //bind the texture to it’s array
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE); //set texture environment parameters

    //here we are setting what textures to use and when. The MIN filter is which quality to show
    //when the texture is near the view, and the MAG filter is which quality to show when the texture
    //is far from the view.

    //The qualities are (in order from worst to best)
    //GL_NEAREST
    //GL_LINEAR
    //GL_LINEAR_MIPMAP_NEAREST
    //GL_LINEAR_MIPMAP_LINEAR

    //And if you go and use extensions, you can use Anisotropic filtering textures which are of an
    //even better quality, but this will do for now.
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    //Here we are setting the parameter to repeat the texture instead of clamping the texture
    //to the edge of our shape. 
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    //Generate the texture
    cv::Mat image_flipped;
    cv::flip(image, image_flipped, 0);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                 image_flipped.cols, image_flipped.rows,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, image_flipped.data);
    glGenerateMipmap(GL_TEXTURE_2D);
    int width, height;
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &width);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &height);
    int max_level = int(std::floor(std::log2(std::max(width, height))));
    glGetTexLevelParameteriv(GL_TEXTURE_2D, max_level, GL_TEXTURE_WIDTH, &width);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, max_level, GL_TEXTURE_HEIGHT, &height);
    assert(width == 1 && height == 1);
    uint8_t average[4];
    glGetTexImage(GL_TEXTURE_2D, max_level, GL_RGBA, GL_UNSIGNED_BYTE, average);
    std::cout << int(average[0]) << " "
              << int(average[1]) << " "
              << int(average[2]) << " "
              << int(average[3]) << std::endl;

    return texture; //return whether it was successfull
}

void FreeTexture( GLuint texture )
{
  glDeleteTextures( 1, &texture );
}

void square (void) {
    glBindTexture(GL_TEXTURE_2D, g_texture); //bind our texture to our shape
    glBegin(GL_QUADS);
    glTexCoord2d(0.0,0.0); glVertex2d(-1.0,-1.0); //with our vertices we have to assign a texcoord
    glTexCoord2d(1.0,0.0); glVertex2d(+1.0,-1.0); //so that our texture has some points to draw to
    glTexCoord2d(1.0,1.0); glVertex2d(+1.0,+1.0);
    glTexCoord2d(0.0,1.0); glVertex2d(-1.0,+1.0);
    glEnd();

//This is how texture coordinates are arranged
//
//  0,1   —–   1,1
//       |     |
//       |     |
//       |     |
//  0,0   —–   1,0

// With 0,0 being the bottom left and 1,1 being the top right.

// Now the point of using the value 0,1 instead of 0,10, is so that it is mapping 1 texture to the
// coordinates. Changing that to 10 would then try to map 10 textures to the one quad. Which because
// I have the repeat parameter set in our texture, it would draw 10 across and 10 down, if we had
// it clamped, we would be still drawing 1. The repeat function is good for things like
// brick walls.
}

void display() {
    glClearColor(0.0,0.0,0.0,1.0);
    glClear(GL_COLOR_BUFFER_BIT);
    glLoadIdentity();
    glEnable(GL_TEXTURE_2D);
    gluLookAt(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    square();
    glutSwapBuffers();
}

cv::Size image_size;

void reshape(int w, int h) {
    glViewport(0, 0, GLsizei(w), GLsizei(h));
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(90, GLfloat(w) / GLfloat(h) *
                       GLfloat(image_size.height) / GLfloat(image_size.width),
                   1.0, 100.0);
    glMatrixMode(GL_MODELVIEW);
}

int main (int argc, char* argv[]) {
    //Load our texture
    cv::Mat image = make_rgba(cv::imread(argv[1]));
    image_size = image.size();

    glutInit (&argc, argv);
    glutInitDisplayMode (GLUT_DOUBLE);
    glutInitWindowSize (image.cols, image.rows);
    glutInitWindowPosition (100, 100);
    glutCreateWindow ("A basic OpenGL Window");
    glutDisplayFunc (display);
    glutIdleFunc (display);
    glutReshapeFunc (reshape);

    GLenum err = glewInit();
    if (err != GLEW_OK) {
      std::cerr << "Error: " << glewGetErrorString(err) << std::endl;
      exit(EXIT_FAILURE);
    }
    std::cerr << "Status: Using GLEW "
              << glewGetString(GLEW_VERSION) << std::endl;

    g_texture = LoadTexture( image );

    glutMainLoop();

    //Free our texture
    FreeTexture(g_texture);

    return EXIT_SUCCESS;
}

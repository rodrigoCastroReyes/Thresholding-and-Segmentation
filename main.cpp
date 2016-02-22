#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cvaux.h>
#include <cxcore.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <list>

using namespace cv;
using namespace std;

/// Global variables
Mat src, src_gray, src_gray_equalize , gradient;
int thresh ;
int max_thresh = 255;

const char* source_window = "Deteccion Lineas";
const char* corners_window = "Corners detected";

/// Function header

Mat applySobel(Mat src_gray);
double findThreshold(IplImage *gris);
void CannyThreshold(Mat input,Mat output);
void applyHoughLineDetection(Mat gray_image);
void applyCornerHarrisDetection(Mat src_gray);
Mat erosion(Mat input);
void drawLine(Mat gray_image,Point p1,Point p2);
void showHistogram(Mat image);

void getLines(Mat gray_image);
bool compareAngle(Vec2f x,Vec2f y);
void findClusters(list <Vec2f> lines,int threshold);

int main( int, char** argv ){
  Mat blur_image;
  int threshold_type=0;
  /// Load source image and convert it to gray
  src = imread(argv[1],CV_LOAD_IMAGE_ANYDEPTH);//leer imagen de 16 bits
  src.convertTo(src_gray,CV_8U);//convertir de 16 bits a 8 bits
  blur(src_gray,blur_image,Size(5,5));//suavizado
  imshow("Original",blur_image);
  //equalizeHist(blur_image, src_gray_equalize );
  gradient = applySobel(blur_image);//encontrar el gradiente de la imagen
  imshow("Realzado",gradient);
  waitKey(0);

  /*gradient = erosion(gradient);
  imshow("Realzado",gradient);
  waitKey(0);*/
  //IplImage temp = gradient;
  //thresh = findThreshold(&temp);
  //threshold(gradient,gradient, thresh, max_thresh,threshold_type );
  threshold(gradient,gradient, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
  
  //bitwise_not(gradient,gradient);
  //CannyThreshold(gradient,gradient);
  imshow("Realzado Binaria",gradient);
  waitKey(0);
  applyHoughLineDetection(gradient);
  getLines(gradient);
  
  /*cornerHarris_demo( 0, 0 );*/
  return(0);
}

Mat erosion(Mat input){
  Mat output = Mat::zeros( input.size(), CV_8UC3 );
  int erosion_type= MORPH_RECT;
  int erosion_size = 5;
  Mat element = getStructuringElement(erosion_type,Size(erosion_size,erosion_size));
  /// Apply the erosion operation
  erode(input, output, element );
  return output;
}


void applyHoughLineDetection(Mat gray_image){
  Mat image;
  cvtColor(src_gray,image ,CV_GRAY2RGB);
  vector<Vec2f> lines;
  float rho,theta;
  int threshold = 100;//numero de puntos intersectados que se consideran que son colineaes
  HoughLines(gray_image, lines, 1, CV_PI/180,threshold,0,0);
  for(int i = 0 ; i< lines.size() ; i++){
    Vec2f v = lines[i];
    rho = v[0];//distancia rho
    theta = v[1];//angulo theta
    if(theta >= 4.0*CV_PI/9.0 && theta <= CV_PI/2.0){//linea vertical
      Point pt3(0,rho/sin(theta));//punto de interseccion de la linea con la primera columna
      Point pt4(gray_image.cols,(rho-gray_image.cols*cos(theta))/sin(theta));//punto de interseccion de la linea con la ultima columna
      line(image,pt3,pt4,Scalar(0,0,255),1);
    }else if(theta >= 0.0 && theta <= CV_PI/9.0 ){
      Point pt1(rho/cos(theta),0);//punto de interseccion de la linea con la primera fila
      Point pt2((rho-gray_image.rows*sin(theta))/cos(theta),gray_image.rows);//punto de interseccion de la linea con la ultima fila
      line(image,pt1,pt2,Scalar(0,0,255),1);
    }
  }
  namedWindow(source_window,WINDOW_AUTOSIZE );
  imshow(source_window,image);
  waitKey(0);
}

void applyCornerHarrisDetection( Mat src_gray ){
  Mat dst, dst_norm, dst_norm_scaled;
  dst = Mat::zeros( src.size(), CV_32FC1 );

  /// Detector parameters
  int blockSize = 5;//tamaño de la vecindad
  int apertureSize = 3;//tamaño del kernel de Sobel: 1,3,5,7
  double k = 0.04;//parametro de formula de harris

  /// Detecting corners
  cornerHarris(src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT );

  /// Normalizing
  normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
  convertScaleAbs( dst_norm, dst_norm_scaled );

  /// Drawing a circle around corners
  for( int j = 0; j < dst_norm.rows ; j++ ){ 
    for( int i = 0; i < dst_norm.cols; i++ ){
        if( (int) dst_norm.at<float>(j,i) > thresh ){
          //circle( dst_norm_scaled, Point( i, j ), 5,  Scalar(0), 2, 8, 0 );
          circle(src_gray,Point(i,j),5,Scalar(0),2,8,0);
        }
    }
  }
  /// Showing the result
  namedWindow( corners_window, WINDOW_AUTOSIZE );
  imshow(corners_window,src_gray);
  waitKey(0);
}

bool compareAngle(Vec2f x,Vec2f y){
  float angle_x,angle_y;
  angle_x = x[1]*(180/CV_PI);
  angle_y = y[1]*(180/CV_PI);

  if(angle_x > angle_y){
    return true;
  }

  return false;
}

void getLines(Mat gray_image){
  Mat image;
  float theta,rho;
  int threshold = 100;//numero de puntos intersectados que se consideran que son colineales
  int indexVertical = 5,indexHorizontal = 5;
  vector<Vec2f> lines;
  Vec2f currentLine;

  cvtColor(src_gray,image ,CV_GRAY2RGB);

  HoughLines(gray_image,lines, 1, CV_PI/180,threshold,0,0);
  
  list <Vec2f> linesVerticalDetected;
  list <Vec2f> linesHorizontalDetected;

  for(int i = 0 ; i< lines.size() ; i++){
    currentLine = lines[i];
    rho = currentLine[0];//distancia rho
    theta = currentLine[1];//angulo theta
    if(theta >= 4.0*CV_PI/9.0 && theta <= CV_PI/2.0){//linea horizontal
      linesHorizontalDetected.push_back(currentLine);
    }else if(theta >= 0.0 && theta <= CV_PI/18.0 ){//linea vertical
      linesVerticalDetected.push_back(currentLine);
    }
  }
  linesHorizontalDetected.sort(compareAngle);
  linesVerticalDetected.sort(compareAngle);

  printf("cols: %d\n",gray_image.cols);
  printf("filas: %d\n",gray_image.rows);

  while (!linesVerticalDetected.empty()){
    currentLine = linesVerticalDetected.back();
    linesVerticalDetected.pop_back();
    rho = currentLine[0];
    theta = 0;
    Point pt1(rho/cos(theta),0);//punto de interseccion de la linea con la primera fila
    Point pt2((rho-gray_image.rows*sin(theta))/cos(theta),gray_image.rows);//punto de interseccion de la linea con la ultima fila
    line(image,pt1,pt2,Scalar(0,0,255),1);
    indexVertical--;
  }

  while (!linesHorizontalDetected.empty()){
    currentLine = linesHorizontalDetected.front();
    linesHorizontalDetected.pop_front();
    rho = currentLine[0];
    theta = CV_PI/2.0;
    Point pt3(0,rho/sin(theta));//punto de interseccion de la linea con la primera columna
    Point pt4(gray_image.cols,(rho-gray_image.cols*cos(theta))/sin(theta));//punto de interseccion de la linea con la ultima columna
    line(image,pt3,pt4,Scalar(0,255,0),1);
    indexHorizontal--;
  }

  namedWindow(source_window,WINDOW_AUTOSIZE );
  imshow(source_window,image);
  waitKey(0);

}
/**
 * @function cornerHarris_demo
 * @brief Executes the corner detection and draw a circle around the possible corners
 */


Mat applySobel(Mat src_gray){
    Mat kernel_x,kernel_y,dst_x,dst_y;
    Mat gradiente;
    Point anchor;
    double delta;
    int scale = 1;
    int ddepth, kernel_size;
    anchor = Point( -1, -1 );
    delta = 0;
    ddepth = -1;//mismos bits que imagen fuente
    
    //Definiendo máscaras de convolución
    //kernel_x = (Mat_<double>(3,3) <<-1,-2,-1,0,0,0,1,2,1);
    //kernel_y = (Mat_<double>(3,3) <<-1,0,1,-2,0,2,-1,0,1);
    //kernel_x = (Mat_<double>(5,5) <<1,2,0,-2,-1,4,8,0,-8,-4,6,12,0,-12,-6,4,8,0,-8,-4,1,2,0,-2,-1);
    //kernel_y = (Mat_<double>(5,5) <<-1,-4.-6,-4,-1,-2,-8,-12,-8,-2,0,0,0,0,0,2,8,12,8,2,1,4,6,4,1);

    /// Apply filter
    /*filter2D(src_gray, dst_x, ddepth , kernel_x, anchor, delta, BORDER_DEFAULT );
    convertScaleAbs(dst_x,dst_x);

    filter2D(src_gray, dst_y, ddepth , kernel_y, anchor, delta, BORDER_DEFAULT );
    convertScaleAbs(dst_y,dst_y);

    add(dst_x,dst_y,gradiente);
    
    bitwise_not(gradiente,gradiente);*/
    /// Generate grad_x and grad_y
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    /// Gradient X
    Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
    //Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );
    /// Gradient Y
    Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
    //Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );
    
    add(abs_grad_x,abs_grad_y,gradiente);
    //addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, gradiente);
    return gradiente;
}

void CannyThreshold(Mat input,Mat output){
   int lowThreshold=0;
   int hightThreshold=255;
   int kernel_size=3;
  /// Canny detector
   Canny(output,output, lowThreshold,hightThreshold , kernel_size );
  /// Using Canny's output as a mask, we display our result
}

double findThreshold(IplImage *gris){
   CvHistogram* hist;
   int hsize[] = {256}; //arreglo de tamaños de todos los histogramas
   float xrange[] = {0.0, 255.0}; // rango
   float * ranges[] = {xrange}; //rangos de todos los histrogramas
   hist = cvCreateHist( 1, hsize, CV_HIST_ARRAY, ranges,1);
   cvCalcHist( &gris, hist, 0, NULL); //calculo el histograma
   double t;
   
    //variables para obtener el umbral
   double temp = 0; //umbral y umbral temporal
   double u1, uo; //medias
   CvScalar s1, s2, s3,s4;//escalares
   printf("x:%d\n",gris->width); 
   printf("y:%d\n",gris->height); 
   s1 = cvGet2D(gris, 0, 0); //esquina superior izquierda
   s2 = cvGet2D(gris, 0, gris->width -1); //esquina superior derecha
   s3 = cvGet2D(gris, gris->height-1,0); //esquina inferior izquierda
   s4 = cvGet2D(gris, gris->height-1, gris->width -1); //esquina inferior derecha
   
   //calculo de Uo inicial
   uo = (double)(s1.val[0] + s2.val[0] + s3.val[0] + s4.val[0])/4;
   printf("\nmedia u0 %f",uo );
   //calculo la media U1
   double sum =0.0, nElements = 0;

   for (int i = 0; i< 256; i++){
      float valor = cvQueryHistValue_1D(hist,i);
      sum+= i*valor;
   }
   int n = (gris->width)*(gris->width);
   u1 = sum/n;

   printf("\nmedia u1 %f", u1);
   //t inicial
   t= (u1 + uo)/2;
   int tround = cvRound(t);

   while(t != temp){
      t = temp;
      sum =0;
      for (int i = 0; i< tround; i++){
         float valor = cvQueryHistValue_1D(hist,i);
         nElements +=valor; 
         sum+= i*valor;
      }
      uo = sum/nElements;
      sum =0;
      nElements = 0;
      for (int i = tround; i< 256; i++){
         float valor = cvQueryHistValue_1D(hist,i);
         nElements+= valor;
         sum+= i*valor;
      }
      u1 = sum/nElements;
      temp = (u1 + uo)/2;
      tround = cvRound(temp);
      printf("\ntemp: %f", temp);
   }
   cvClearHist(hist);
   return t;
}

void showHistogram(Mat image){
   /// Establish the number of bins
  int histSize = 256;
  /// Set the ranges ( for B,G,R) )
  float range[] = { 0, 256 } ;
  const float* histRange = { range };
  bool uniform = true; bool accumulate = false;
  Mat hist;
  /// Compute the histograms:
  calcHist(&image, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );
  // Draw the histograms for B, G and R
  int hist_w = 512; int hist_h = 400;
  int bin_w = cvRound( (double) hist_w/histSize );
  Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

  /// Normalize the result to [ 0, histImage.rows ]
  normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
  /// Draw for each channel
  for( int i = 1; i < histSize; i++ ){
    line( histImage,Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ) ,
                    Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
                    Scalar( 255, 0, 0), 2, 8, 0);
  }
  namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
  imshow("calcHist Demo", histImage );
  waitKey(0);

}
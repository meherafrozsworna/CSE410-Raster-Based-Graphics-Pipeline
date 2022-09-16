#include <iostream>
#include <fstream>
#include <string>
#include <stack>
#include <cmath>
#include <vector>
#include <cstdio>
#include <cmath>
#include "bitmap_image.hpp"

#define PI (2*acos(0.0))

using namespace std;

struct point{
    double x,y,z;
};
struct Vector{
    double x,y,z ;
};
struct Matrix{
    double matrix[4][4];
};
struct Triangle{
    point points[3];
    int color[3];
};
class Color
{
public:
    int color[3];
    Color(){}
    Color(int r, int g, int b)
    {
        color[0] = r ;
        color[1] = g ;
        color[2] = b ;
    }

    void print(){
        cout << "color : " << "r : " << color[0]
        << "g : " << color[1]<< " b : "<< color[2] << endl;
    }
};
stack<Matrix> stack_matrix ;
vector<int> instruction_after_push_count ;
vector<Triangle> triangleList ;
int push_count = 0 ;
FILE *fp;
double** z_buffer ;
Color** image_color ;

void print_vector(Vector v){
    cout << "vector : " << v.x << "  " << v.y << "  " << v.z << endl;
}
void print_point(point p){
    cout << "vector : " << p.x << "  " << p.y << "  " << p.z << endl;
}
void print_matrix(Matrix a){
    cout << endl << "Matrix: " << endl;
    for(int i = 0; i < 4; ++i){
        for(int j = 0; j < 4; ++j){
            cout << " " << a.matrix[i][j];
            if(j == 3)
                cout << endl;
        }
    }
}
void print_point_matrix(double a[4][1]){
    //cout << endl << "Point Matrix: " << endl;
    int r = 3 , c = 1 ;
    for(int i = 0; i < r; ++i)
        for(int j = 0; j < c; ++j){
            //printf("%.7lf  ",a[i][j]);
            fprintf(fp, "%.7lf ", a[i][j]);
        }
    //cout << endl;
    fprintf(fp,"\n");

}
Matrix matrix_matrix_multiplication(Matrix mat1 , Matrix mat2 )
{
    Matrix result ;

    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
            result.matrix[i][j] = 0;
            for(int k = 0; k < 4; k++){
                result.matrix[i][j] += mat1.matrix[i][k] * mat2.matrix[k][j];
            }
        }
    }
    return result;
}
void matrix_point_multiplication(double a[4][4], double b[4][1],double mult[4][1])
{
    int r1 = 4 , c1 = 4 , r2 = 4 , c2 = 1;

    for(int i = 0; i < r1; i++)
        for(int j = 0; j < c2; j++){
            mult[i][j]=0;
        }

    for(int i = 0; i < r1; i++){
        for(int j = 0; j < c2; j++){
            for(int k = 0; k < c1; k++){
                mult[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

double dot_product(Vector a, Vector b)
{
    double product = 0;
    product += a.x * b.x;
    product += a.y * b.y;
    product += a.z * b.z;
    return product;
}

Vector cross_product(Vector a, Vector b)
{
    Vector result ;
    result.x = a.y * b.z - a.z * b.y;
    result.y = a.z * b.x - a.x * b.z;
    result.z = a.x * b.y - a.y * b.x;
    return result;
}
Vector rotation(Vector x , Vector a , double angle){
    Vector result ;
    double dot = dot_product(a,x);
    Vector cross = cross_product(a,x);
    double costheta = cos((angle*PI)/180);
    double sintheta = sin((angle*PI)/180);

//    cout << "a matrix : "<<  a.x <<" "<< a.y <<" "<< a.z << endl;
//    cout << "dot : " << dot << endl;
//    cout << "cross : " << cross.x << "  " << cross.y << "  " << cross.z << endl;
//    cout << "cos sin " << costheta << " " << sintheta << endl;

    result.x = costheta * x.x + (1-costheta) * dot * a.x + sintheta * cross.x ;
    result.y = costheta * x.y + (1-costheta) * dot * a.y + sintheta * cross.y ;
    result.z = costheta * x.z + (1-costheta) * dot * a.z + sintheta * cross.z ;

    cout << "result in rotation ";
//    cout << result.x << " " << result.y << "  " << result.z << endl;
    return result;
}

Vector normalize(Vector a)
{
    double mod_a = sqrt(a.x*a.x + a.y*a.y + a.z*a.z);
    Vector a_norm ;
    a_norm.x = a.x / mod_a ;
    a_norm.y = a.y / mod_a ;
    a_norm.z = a.z / mod_a ;
    return a_norm ;
}

void view_transformation(point eye, point look , point up)
{
   Vector l,r,u ;
    l.x = look.x - eye.x ;
    l.y = look.y - eye.y ;
    l.z = look.z - eye.z ;
    l = normalize(l);
    Vector up_vec ;
    up_vec.x = up.x  ;
    up_vec.y = up.y ;
    up_vec.z = up.z ;
    r = cross_product(l,up_vec);
    r = normalize(r);
    u = cross_product(r,l);

    Matrix T , R , V;

    for (int i = 0; i < 4 ; ++i) {
        for (int j = 0; j < 4 ; ++j) {
            if (i==j){
                T.matrix[i][j] = 1;
            }else{
                T.matrix[i][j] = 0;
            }
        }
    }
    T.matrix[0][3] = -eye.x ;
    T.matrix[1][3] = -eye.y ;
    T.matrix[2][3] = -eye.z ;

    R.matrix[0][0] = r.x , R.matrix[0][1] = r.y , R.matrix[0][2] = r.z , R.matrix[0][3] = 0.0 ;
    R.matrix[1][0] = u.x , R.matrix[1][1] = u.y , R.matrix[1][2] = u.z , R.matrix[1][3] = 0.0 ;
    R.matrix[2][0] = -l.x , R.matrix[2][1] = -l.y , R.matrix[2][2] = -l.z , R.matrix[2][3] = 0.0 ;
    R.matrix[3][0] = 0.0  , R.matrix[3][1] = 0.0  , R.matrix[3][2] = 0.0  , R.matrix[3][3] = 1.0 ;

    V = matrix_matrix_multiplication(R,T);

    fstream inputfile;
    inputfile.open("stage1.txt");
    fp = fopen("stage2.txt","w");
    if (inputfile.is_open()) {
        point p ;
        int c ;
        while (inputfile >> p.x >> p.y >> p.z){
            c++ ;
            double point_matrix[4][1] , result[4][1];
            point_matrix[0][0] = p.x ;
            point_matrix[1][0] = p.y ;
            point_matrix[2][0] = p.z ;
            point_matrix[3][0] = 1.0 ;

            matrix_point_multiplication(V.matrix,point_matrix,result);
            print_point_matrix(result);
            if (c == 3){
                fprintf(fp,"\n");
                c = 0;
            }
        }
    }
    fclose(fp);
    inputfile.close();
}

void projection_transformation(double fovY , double aspect_ratio, double near, double far)
{
    //cout <<endl<<  fovY << " " << aspect_ratio << " " << near << " " << far << endl;
    double fovX = fovY * aspect_ratio ;
    double t = near * tan(((fovY/2.0)*PI)/180.0);
    double r = near * tan(((fovX/2.0)*PI)/180.0);
//    double t = near * tan(fovY/2.0);
//    double r = near * tan(fovX/2.0);

    Matrix P ;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4 ; ++j) {
            P.matrix[i][j] = 0;
        }
    }
    P.matrix[0][0] = near/r ;
    P.matrix[1][1] = near/t ;
    P.matrix[2][2] = -(far + near)/(far - near) ;
    P.matrix[2][3] = -(2 * far * near)/(far - near) ;
    P.matrix[3][2] = -1 ;

    fstream inputfile;
    inputfile.open("stage2.txt");
    fp = fopen("stage3.txt","w");
    if (inputfile.is_open()) {
        point p ;
        int c ;
        while (inputfile >> p.x >> p.y >> p.z){
            c++ ;
            double point_matrix[4][1] , result[4][1];
            point_matrix[0][0] = p.x ;
            point_matrix[1][0] = p.y ;
            point_matrix[2][0] = p.z ;
            point_matrix[3][0] = 1.0 ;

            matrix_point_multiplication(P.matrix,point_matrix,result);
            for (int i = 0; i < 4 ; ++i) {
                result[i][0] = result[i][0]/result[3][0];
            }
            print_point_matrix(result);
            if (c == 3){
                fprintf(fp,"\n");
                c = 0;
            }
        }
    }
    fclose(fp);
    inputfile.close();
}

double distance(double x1, double y1 , double x2, double y2){
    return sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2));
}

void z_buffer_algorithm()
{
    fstream inputfile;
    inputfile.open("stage3.txt");
    if (inputfile.is_open()) {
        point p;
        while (inputfile >> p.x >> p.y >> p.z) {
            Triangle triangle;
            triangle.points[0] = p;
            inputfile >> p.x >> p.y >> p.z;
            triangle.points[1] = p;
            inputfile >> p.x >> p.y >> p.z;
            triangle.points[2] = p;
            for (int i = 0; i < 3; ++i) {
                triangle.color[i] = (rand() % 255);
            }
            triangleList.push_back(triangle);
        }
    }
    inputfile.close();

    fstream confile;
    int screen_width , screen_height ;
    double left_x , right_x , bottom_y , top_y ,front_z , rear_z ;
    confile.open("config.txt");
    if (confile.is_open()) {
        confile >> screen_width >> screen_height ;
        confile >> left_x >> bottom_y >> front_z >> rear_z ;
        right_x = -left_x ;
        top_y = -bottom_y ;
    }
    confile.close();
    cout << "\nList size : "<<  triangleList.size() << endl;

    z_buffer = new double *[screen_height];
    image_color = new Color*[screen_height];

    for(int i = 0; i < screen_height; ++i) {
        z_buffer[i] = new double [screen_width];
        image_color[i] = new Color[screen_width];
        for (int j = 0; j < screen_width ; ++j) {
            z_buffer[i][j] = rear_z ;
            Color color(0,0,0);
            image_color[i][j] = color ;
        }
    }

    double dx = (right_x - left_x)/screen_width ;
    double dy = (top_y - bottom_y)/screen_height ;
    double Top_Y = top_y -dy/2 ;
    double Left_X = left_x + dx/2 ;
    double Bottom_Y = bottom_y + dy/2 ;
    double Right_X = right_x - dx/2 ;

    for (int i = 0; i < triangleList.size() ; ++i) {
        cout << i<< endl;
        Triangle t = triangleList[i];
        //triangleList.pop_back() ;
        //clip
        double max_y = -999999.0 ;
        double min_y = 999999.0 ;
        double max_x = -999999.0 ;
        double min_x = 99999.0 ;

        for (int j = 0; j < 3 ; ++j) {
            //cout << "t.points[j].y : " << t.points[j].y << endl;
            if (max_y < t.points[j].y ){
                //cout <<"at max "<< "t.points[j].y : " << t.points[j].y << endl;
                max_y = t.points[j].y ;
            }
            if (min_y > t.points[j].y){
                min_y = t.points[j].y ;
            }
            if (max_x < t.points[j].x){
                max_x = t.points[j].x ;
            }
            if (min_x > t.points[j].x){
                min_x = t.points[j].x ;
            }
        }
        if (max_y > Top_Y){
            max_y = Top_Y ;
        }
        if (min_y < Bottom_Y){
            min_y = Bottom_Y ;
        }
        if (max_x > Right_X){
            max_x = Right_X ;
        }
        if (min_x < Left_X){
            min_x = Left_X ;
        }

        int row_top = round((Top_Y - max_y)/dy) ; //round baad disi
        int row_bottom = round((Top_Y - min_y)/dy) ;

        if(row_top < 0){
            row_top = 0;
        }
        if(row_bottom > screen_height-1){
            row_bottom = screen_height-1;
        }

        int count = 0;
        while (row_top+count <= row_bottom) {
            int row = row_top + count;
            //cout << "rowstart " << row_top << "  " << "row end " << row_bottom << " "<< row << endl;

            double y_scanline = Top_Y - row * dy;

            vector<double> intersect;
            vector<double> z_intersection;
            for (int j = 0; j < 3; ++j) {
                double mx_x = (t.points[j].x > t.points[(j + 1) % 3].x) ? (t.points[j].x) : t.points[(j + 1) % 3].x;
                double mn_x = (t.points[j].x < t.points[(j + 1) % 3].x) ? (t.points[j].x) : t.points[(j + 1) % 3].x;


                    double x = ((y_scanline - t.points[j].y)
                                * (t.points[j].x - t.points[(j + 1) % 3].x)) /
                               (t.points[j].y - t.points[(j + 1) % 3].y) +
                               t.points[j].x;

                    double z = ((y_scanline - t.points[j].y)
                                * (t.points[(j + 1) % 3].z - t.points[j].z)) /
                               (t.points[(j + 1) % 3].y - t.points[j].y) +
                               t.points[j].z;

                    double xa1 = distance(t.points[j].x,t.points[j].y,x,y_scanline);
                    double xa2 = distance(t.points[(j+1)%3].x,t.points[(j+1)%3].y,x,y_scanline);
                    double x12 =
                            distance(t.points[j].x,t.points[j].y,t.points[(j+1)%3].x,t.points[(j+1)%3].y);
//                if (x <= mx_x && x >= mn_x) {
//                        intersect.push_back(x);
//                        z_intersection.push_back(z);
//                    }
                if (xa1+xa2-x12 < 0.0001){
                    intersect.push_back(x);
                    z_intersection.push_back(z);
                }


            }
            //cout << "size of intersect : "<< intersect.size() << endl;

            if (intersect.size() > 2) {
                //cout << "AAAAAAAAAAAAAAA" << endl;
                int freq = std::count(intersect.begin(), intersect.end(), intersect[0]);
                //int freq2 = std::count(intersect.begin(), intersect.end(), intersect[1]);
                if (freq > 1) {
                    intersect.erase(intersect.begin() + 0);
                    z_intersection.erase(z_intersection.begin() + 0);
                } else {
                    intersect.erase(intersect.begin() + 1);
                    z_intersection.erase(z_intersection.begin() + 1);
                }
            }

            if (intersect.size() == 2){
                double xa, xb, za, zb;
                if (intersect[0] < intersect[1]) {
                    xa = intersect[0];
                    xb = intersect[1];
                    za = z_intersection[0];
                    zb = z_intersection[1];
                } else {
                    xa = intersect[1];
                    xb = intersect[0];
                    za = z_intersection[1];
                    zb = z_intersection[0];
                }

                if (xa < Left_X) {
                    xa = Left_X;
                }
                if (xb > Right_X) {
                    xb = Right_X;
                }


                int col_left = round((xa - Left_X) / dx);
                int col_right = round((xb - Left_X) / dx);

                double xp = Left_X + col_left * dx;
                double zp = za + ((xp - xa) * (zb - za)) / (xb - xa);

                int col_count = 0;
                while (col_right >= col_left + col_count) {
                    int col = col_left + col_count;
                    double z = zp + col_count * dx * (zb - za) / (xb - xa);
                    //cout << z << endl;
                    if (z_buffer[row][col] > z && z > front_z) {
                        z_buffer[row][col] = z;
                        Color new_color(triangleList[i].color[0],
                                        triangleList[i].color[1],
                                        triangleList[i].color[2]);
                        image_color[row][col] = new_color;
                    }
                    col_count++;
                }
            }
            count++;
        }
    }

    bitmap_image image(screen_width,screen_height);

    for (int i = 0; i < screen_height ; ++i) {
        for (int j = 0; j < screen_width ; ++j) {
            image.set_pixel(j,i,image_color[i][j].color[0],
                            image_color[i][j].color[1],
                            image_color[i][j].color[2]);
        }
    }
    image.save_image("out.bmp");

    fp = fopen("z_buffer.txt","w");
    for (int i = 0; i < screen_height ; ++i) {
        for (int j = 0; j < screen_width ; ++j) {
            //fprintf(fp,"%.7lf\t",z_buffer[i][j]);
            if (z_buffer[i][j] < rear_z){
                fprintf(fp,"%.6lf\t",z_buffer[i][j]);
            }
        }
        fprintf(fp,"\n");
    }

    for( int i = 0 ; i < screen_height ; i++ ){
        delete[] z_buffer[i];
        delete[] image_color[i];
    }

    delete[] z_buffer;
    delete[] image_color;

    fclose(fp);
}

int main() {
    point eye, look,up ;
    double  fovY , aspect_ratio , near , far ;

    Matrix mat ;
    for (int i = 0; i < 4 ; ++i) {
        for (int j = 0; j < 4 ; ++j) {
            if (i == j){
                mat.matrix[i][j] = 1.0;
            }else{
                mat.matrix[i][j] = 0.0;
            }
        }
    }
    stack_matrix.push(mat);

    fp = fopen("stage1.txt","w");
    fstream scenefile;
    scenefile.open("scene.txt");
    if (scenefile.is_open()) {
        scenefile >> eye.x >> eye.y >> eye.z ;
        scenefile >> look.x >> look.y >> look.z ;
        scenefile >> up.x >> up.y >> up.z ;
        scenefile >> fovY >> aspect_ratio >> near >> far;

        string input_command ;
        while (true){
            scenefile >> input_command ;
            if (input_command == "triangle"){
                point triangle[3] ;
                for (int i = 0; i < 3 ; ++i) {
                    scenefile >> triangle[i].x >> triangle[i].y >> triangle[i].z ;
                }

                //cout << "Triangle : " << endl;
                for (int i = 0; i < 3 ; ++i) {
                    double point_matrix[4][1] , new_point_matrix[4][1] ;
                    point_matrix[0][0] = triangle[i].x ;
                    point_matrix[1][0] = triangle[i].y ;
                    point_matrix[2][0] = triangle[i].z ;
                    point_matrix[3][0] = 1 ;

                    matrix_point_multiplication(stack_matrix.top().matrix,point_matrix,new_point_matrix);
                    print_point_matrix(new_point_matrix);
                }
                fprintf(fp,"\n");


            }else if(input_command == "translate"){
                Matrix mat ;
                double tx,ty,tz ;
                scenefile >> tx >> ty >> tz ;
                for (int i = 0; i < 4 ; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        if (i == j){
                            mat.matrix[i][j] = 1;
                        }else{
                            mat.matrix[i][j] = 0;
                        }
                    }
                }
                mat.matrix[0][3] = tx;
                mat.matrix[1][3] = ty;
                mat.matrix[2][3] = tz;

                Matrix result = matrix_matrix_multiplication(stack_matrix.top(),mat);
                stack_matrix.push(result);
//                cout << "after translate " << endl;
//                print_matrix(result);

                if (push_count > 0){
                    instruction_after_push_count.back()++ ;
                }

            }else if(input_command == "scale"){
                Matrix mat ;
                double sx, sy, sz ;
                scenefile >> sx >> sy >> sz ;
                for (int i = 0; i < 4 ; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        mat.matrix[i][j] = 0;
                    }
                }
                mat.matrix[0][0] = sx;
                mat.matrix[1][1] = sy;
                mat.matrix[2][2] = sz;
                mat.matrix[3][3] = 1;

                Matrix result = matrix_matrix_multiplication(stack_matrix.top(),mat);
//                print_matrix(result);
                stack_matrix.push(result);
                if (push_count > 0){
                    instruction_after_push_count.back()++ ;
                }

            }else if(input_command == "rotate"){
                double angle , ax, ay , az ;
                scenefile >> angle >> ax >> ay >> az ;
                double mod_a = sqrt(ax*ax + ay*ay + az*az);
                Vector a ;
                a.x = ax / mod_a ;
                a.y = ay / mod_a ;
                a.z = az / mod_a ;
                Vector i_vector,j_vector,k_vector ;
                i_vector.x = 1 , i_vector.y = 0 ; i_vector.z = 0;
                j_vector.x = 0 , j_vector.y = 1 ; j_vector.z = 0;
                k_vector.x = 0 , k_vector.y = 0 ; k_vector.z = 1;
                Vector c1 = rotation(i_vector,a,angle);
                Vector c2 = rotation(j_vector,a,angle);
                Vector c3 = rotation(k_vector,a,angle);

//                cout << "debug "<< c1.x << "  " << c1.y << "  " << c1.z << endl;
                Matrix rotation_matrix ;
                rotation_matrix.matrix[0][0] = c1.x , rotation_matrix.matrix[0][1] = c2.x , rotation_matrix.matrix[0][2] = c3.x , rotation_matrix.matrix[0][3] = 0 ;
                rotation_matrix.matrix[1][0] = c1.y , rotation_matrix.matrix[1][1] = c2.y , rotation_matrix.matrix[1][2] = c3.y , rotation_matrix.matrix[1][3] = 0 ;
                rotation_matrix.matrix[2][0] = c1.z , rotation_matrix.matrix[2][1] = c2.z , rotation_matrix.matrix[2][2] = c3.z , rotation_matrix.matrix[2][3] = 0 ;
                rotation_matrix.matrix[3][0] = 0.0    , rotation_matrix.matrix[3][1] = 0.0    , rotation_matrix.matrix[3][2] = 0.0    , rotation_matrix.matrix[3][3] = 1.0 ;

//                cout << "before rotation "<< endl;
//                print_matrix(rotation_matrix);
                Matrix result = matrix_matrix_multiplication(stack_matrix.top(),rotation_matrix);
                stack_matrix.push(result);
//                cout << "rotation matrix : " << endl;
//                print_matrix(result);
                if (push_count > 0){
                    instruction_after_push_count.back()++ ;
                }

            }else if(input_command == "push"){
                push_count++ ;
                instruction_after_push_count.push_back(0);
                // ekta push er por koyta r/s/t stack a dhukse oita count korbo,
                // pop korle totota stack theke ber korbo ,
                // count ++ for all the value in the vector
                // pop pele last vector er count poriman shob gula theke minus hobe and toto gula pop hobe

            }else if(input_command == "pop"){
                if (instruction_after_push_count.empty()){
                    cout << "pop comes without push , error !!!! ";
                }else{
                    int count = instruction_after_push_count.back();
                    while (count > 0){
                        stack_matrix.pop();
                        count-- ;
                    }
                }
            }else if(input_command == "end"){
                break;
            }

        }
        scenefile.close();
    }
    fclose(fp);

    view_transformation(eye,look,up);
    projection_transformation(fovY,aspect_ratio,near,far);
    z_buffer_algorithm();
    return 0;
}

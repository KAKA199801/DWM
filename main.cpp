#include "BpNet.h"
#include<iostream>
#include<fstream>
#include<cmath>
using namespace std;
int main()
{  //样本文件
    const char* filename="data.txt";
    BpNet bp;
    bp.init();
    int times=0;
    while(bp.error>0.1123)
    {
        bp.e=0.0;
        times++;
        bp.train(filename);
        cout <<"Times="<<times<<" error=" <<bp.error<<endl;
    }
    //cout << "trainning complete..." << endl;
    bp.writetrain();//保存网络
    //识别
    bp.readtrain();//读入网络
    const char*filename1="test.txt";
    const char*filename2 ="Result.txt";
    bp.recognize(filename1,filename2);
    return 0;
}

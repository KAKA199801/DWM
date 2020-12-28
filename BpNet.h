#ifndef BPNET_H
#define BPNET_H
#include<iostream>
#include<fstream>
#include<cmath>
using namespace std;
#define  innode 16  //输入结点数
#define  hidenode 8//隐含结点数
#define  outnode 1 //输出结点数
class BpNet
{
public:
    //FILE *fdata;
    //const char*filename;
    void train(const char*filename);//Bp训练
    //double p[innode];     //输入的一个样本
    //double t[outnode];    //样本要输出的值
    void recognize(const char*filename, const char*filename1);//Bp识别  filename中数据集，结果保存到filename1
    void writetrain(); //写训练完的权值
    void readtrain(); //读训练好的权值
    BpNet();
    virtual ~BpNet();
public:
    void init();
    double w[innode][hidenode];//隐含结点权值
    double w1[hidenode][outnode];//输出结点权值
    double b1[hidenode];//隐含结点阀值
    double b2[outnode];//输出结点阀值
    double rate_w; //权值学习率（输入层-隐含层)
    double rate_w1;//权值学习率 (隐含层-输出层)
    double rate_b1;//隐含层阀值学习率
    double rate_b2;//输出层阀值学习率
    double e;//误差计算
    double error;//允许的最大误差
    double result[outnode];// Bp输出
};
#endif // BPNET_H

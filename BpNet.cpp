#include "BpNet.h"
#include<iostream>
#include<fstream>
#include<cmath>
#include<cstdio>
#include<cstdlib>
BpNet::BpNet()
{
    error = 10;
    e = 0.0;
    rate_w = 0.2;  //权值学习率（输入层--隐含层)
    rate_w1 = 0.2; //权值学习率 (隐含层--输出层)
    rate_b1 = 0.2; //隐含层阀值学习率
    rate_b2 = 0.2; //输出层阀值学习率
}
BpNet::~BpNet()
{
}
void winit(double w[], int n) //权值初始化
{
    for (int i = 0; i < n; i++)
        w[i] = (2.0*(double)rand() / RAND_MAX) - 1;
        //w[i] = 1 / n;
}
void BpNet::init()
{
    winit((double*)w, innode*hidenode);
    winit((double*)w1, hidenode*outnode);
    winit(b1, hidenode);
    winit(b2, outnode);
}
void BpNet::train(const char *filename)
{
    double pp[hidenode];//隐含结点的校正误差
    double qq[outnode];//希望输出值与实际输出值的偏差
    double yd[outnode];//希望输出值
    double x[innode]; //输入向量
    double x1[hidenode];//隐含结点状态值
    double x2[outnode];//输出结点状态值
    double o1[hidenode];//隐含层激活值
    double o2[hidenode];//输出层激活值
    //判断文件读入
    FILE *fp = fopen(filename, "rt");
    if (fp == NULL)//文件打开失败:返回
    {
        exit(1);
    }
    const char *xx2="xx.txt";
    FILE * ff = fopen("xx.txt", "w+");
    while (!feof(fp)){
            double sample;
            int colums = 31;
            int i = 0;
            int j = 0;
            for (int c = 0; c < colums; c++){
                fscanf(fp, "%lf", &sample);
                //cout << sample << " ";
                if (c < outnode&&i < outnode)
                    yd[i++] = sample;//期望输出的样本
                else
                    if (j < innode)
                        x[j++] = sample; //输入的样本
            }

//前向传播之计算隐含层各单元的输入值和经过激活函数后的输出
        //构造每个样品的输入和输出标准
        for (int j = 0; j<hidenode; j++)
        {
            o1[j] = 0.0;
            for (int i = 0; i<innode; i++)
                o1[j] = o1[j] + w[i][j] * x[i];//隐含层各单元输入激活值
            x1[j] = 1.0 / (1 + exp(-o1[j] - b1[j]));//隐含层各单元的输出
            //    if(o1[j]+b1[j]>0) x1[j]=1;
            //else x1[j]=0;

        }
//前向传播之计算输出层各单元的输入值和经过激活函数后的输出
        for (int k = 0; k<outnode; k++)
        {
            o2[k] = 0.0;
            for (int j = 0; j<hidenode; j++)
                o2[k] = o2[k] + w1[j][k] * x1[j]; //输出层各单元输入激活值
            x2[k] = 1.0 / (1.0 + exp(-o2[k] - b2[k])); //输出层各单元输出
            fprintf(ff, "%lf\n", x2[k]);
            //    if(o2[k]+b2[k]>0) x2[k]=1;
            //    else x2[k]=0;
        }
//后向传播之修正隐含层与输出层间的权值
        for (int k = 0; k<outnode; k++)
        {
            qq[k] = (yd[k] - x2[k])*x2[k] * (1 - x2[k]); //希望输出与实际输出的偏差
            for (int j = 0; j<hidenode; j++)
                w1[j][k] += rate_w1*qq[k] * x1[j];  //下一次的隐含层和输出层之间的新连接权
        }
//后向传播之修正隐含层与输入层间的权值
        for (int j = 0; j<hidenode; j++)
        {
            pp[j] = 0.0;
            for (int k = 0; k<outnode; k++)
                pp[j] = pp[j] + qq[k] * w1[j][k];
            pp[j] = pp[j] * x1[j] * (1 - x1[j]); //隐含层的校正误差
            for (int i = 0; i<innode; i++)
                w[i][j] += rate_w*pp[j] * x[i]; //下一次的输入层和隐含层之间的新连接权
        }
//计算误差
        for (int k = 0; k<outnode; k++)
        {
            e += fabs(yd[k] - x2[k])*fabs(yd[k] - x2[k]); //计算均方差
        }
        error = e / 2.0;
//修正隐含层和输出层之间、输入层和隐含层之间的偏置参数
        for (int k = 0; k<outnode; k++)
            b2[k] = b2[k] + rate_b2*qq[k]; //下一次的隐含层和输出层之间的新阈值
        for (int j = 0; j<hidenode; j++)
            b1[j] = b1[j] + rate_b1*pp[j]; //下一次的输入层和隐含层之间的新阈值
    }
    fclose(ff);
    fclose(fp);
}
void BpNet::recognize(const char*filename, const char*filename1)
{
    FILE *fp = fopen(filename, "rt");
    if (fp == NULL)exit(1);
    FILE *Rrec;
    Rrec = fopen(filename1, "w+");
    double x[innode]; //输入向量
    double x1[hidenode]; //隐含结点状态值
    double x2[outnode]; //输出结点状态值
    double o1[hidenode]; //隐含层激活值
    double o2[hidenode]; //输出层激活值
    while (!feof(fp)){
        double sample;
        int colums = 30;
        int j = 0;
        for (int c = 0; c < colums; c++){
            fscanf(fp, "%lf", &sample);
            //cout << sample << " ";
                if (j < innode)
                    x[j++] = sample; //输入的样本
        }
    for (int j = 0; j < hidenode; j++)
    {
        o1[j] = 0.0;
        for (int i = 0; i < innode; i++)
            o1[j] = o1[j] + w[i][j] * x[i]; //隐含层各单元激活值
        x1[j] = 1.0 / (1.0 + exp(-o1[j] - b1[j])); //隐含层各单元输出
        //if(o1[j]+b1[j]>0) x1[j]=1;
        //    else x1[j]=0;
    }
    for (int k = 0; k < outnode; k++)
    {
        o2[k] = 0.0;
        for (int j = 0; j < hidenode; j++)
            o2[k] = o2[k] + w1[j][k] * x1[j];//输出层各单元激活值
        x2[k] = 1.0 / (1.0 + exp(-o2[k] - b2[k]));//输出层各单元输出
        //if(o2[k]+b2[k]>0) x2[k]=1;
        //else x2[k]=0;
    }
    for (int k = 0; k < outnode; k++)
    {
        result[k] = x2[k];
    }
    fprintf(Rrec,"%lf\n",result[0]);
  }
    fclose(Rrec);
    fclose(fp);
}
void BpNet::writetrain()
{
    FILE *stream0;
    FILE *stream1;
    FILE *stream2;
    FILE *stream3;
    int i, j;
    //隐含结点权值写入
    if ((stream0 = fopen("w.txt", "w+")) == NULL)
    {
        cout << "创建文件失败!";
        exit(1);
    }
    for (i = 0; i<innode; i++)
    {
        for (j = 0; j<hidenode; j++)
        {
            fprintf(stream0, "%f\n", w[i][j]);
        }
    }
    fclose(stream0);
    //输出结点权值写入
    if ((stream1 = fopen("w1.txt", "w+")) == NULL)
    {
        cout << "创建文件失败!";
        exit(1);
    }
    for (i = 0; i<hidenode; i++)
    {
        for (j = 0; j<outnode; j++)
        {
            fprintf(stream1, "%f\n", w1[i][j]);
        }
    }
    fclose(stream1);
    //隐含结点阀值写入
    if ((stream2 = fopen("b1.txt", "w+")) == NULL)
    {
        cout << "创建文件失败!";
        exit(1);
    }
    for (i = 0; i<hidenode; i++)
        fprintf(stream2, "%f\n", b1[i]);
    fclose(stream2);
    //输出结点阀值写入
    if ((stream3 = fopen("b2.txt", "w+")) == NULL)
    {
        cout << "创建文件失败!";
        exit(1);
    }
    for (i = 0; i<outnode; i++)
        fprintf(stream3, "%f\n", b2[i]);
    fclose(stream3);
}
void BpNet::readtrain()
{
    FILE *stream0;
    FILE *stream1;
    FILE *stream2;
    FILE *stream3;
    int i, j;
    //隐含结点权值读出
    if ((stream0 = fopen("w.txt", "r")) == NULL)
    {
        cout <<"打开文件失败!";
        exit(1);
    }
    float  wx[innode][hidenode];
    for (i = 0; i<innode; i++)
    {
        for (j = 0; j<hidenode; j++)
        {
            fscanf(stream0, "%f", &wx[i][j]);
            w[i][j] = wx[i][j];
        }
    }
    fclose(stream0);
    //输出结点权值读出
    if ((stream1 = fopen("w1.txt", "r")) == NULL)
    {
        cout << "打开文件失败!";
        exit(1);
    }
    float  wx1[hidenode][outnode];
    for (i = 0; i<hidenode; i++)
    {
        for (j = 0; j<outnode; j++)
        {
            fscanf(stream1, "%f", &wx1[i][j]);
            w1[i][j] = wx1[i][j];
        }
    }
    fclose(stream1);
    //隐含结点阀值读出
    if ((stream2 = fopen("b1.txt", "r")) == NULL)
    {
        cout << "打开文件失败!";
        exit(1);
    }
    float xb1[hidenode];
    for (i = 0; i<hidenode; i++)
    {
        fscanf(stream2, "%f", &xb1[i]);
        b1[i] = xb1[i];
    }
    fclose(stream2);
    //输出结点阀值读出
    if ((stream3 = fopen("b2.txt", "r")) == NULL)
    {
        cout << "打开文件失败!";
        exit(1);
    }
    float xb2[outnode];
    for (i = 0; i<outnode; i++)
    {
        fscanf(stream3, "%f", &xb2[i]);
        b2[i] = xb2[i];
    }
    fclose(stream3);
}

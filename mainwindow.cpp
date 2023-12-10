#include "mainwindow.h"
#include <iostream>
#include <fstream>
#include <sys/stat.h>

#include <QIcon>
#include <QDebug>


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    this->setWindowTitle("Robustness Verifier");
    //    this->setWindowIcon(QIcon(":/icons/images/notepad.png"));  //设置应用显示图标

    init_GUI();
}

void MainWindow::init_GUI(){
    // 菜单栏
    fileMenu = new QMenu(this);
    helpMenu = new QMenu(this);
    fileMenu = menuBar()->addMenu(tr( "File" ));
    menuBar()->addSeparator();
    menuBar()->addSeparator();
    helpMenu = menuBar()->addMenu(tr("Help" ));

    // 菜单子节点
    openAct = new QAction(QIcon(":/icons/images/open_file.png"), tr("Open a result data file"), this);
    openAct->setShortcut(tr("Ctrl+O" ));
    openAct->setStatusTip(tr("Open a result data file"));
    saveAct = new QAction(QIcon( ":/icons/images/save_file.png"), tr("Save results"), this);
    saveAct->setShortcut(tr("Ctrl+S"));
    saveAct->setStatusTip(tr("Save results"));
    saveasAct = new QAction(QIcon(":/icons/images/save_as.png"), tr("Save results as"), this);
    saveasAct->setShortcut(tr("Ctrl+Shift+S"));
    saveasAct->setStatusTip(tr("Save results as"));
    aboutQtAct = new QAction(QIcon(":/icons/images/about.png"), tr("About Qt"), this);
    aboutQtAct->setStatusTip(tr("Information about Qt"));

    // 填充菜单栏
    fileMenu->addAction(openAct);
    fileMenu->addAction(saveAct);
    fileMenu->addAction(saveasAct);
    helpMenu->addAction(aboutQtAct);

    this->robustnessView = new RobustnessView();
    this->robustnessView->setObjectName(QString::fromUtf8("robustnessView"));

    this->ui->horizontalLayout_robustness->addWidget(this->robustnessView);

    this->fairnessView = new FairnessView();
    this->fairnessView->setObjectName(QString::fromUtf8("fairnessView"));

    this->ui->horizontalLayout_fairness->addWidget(this->fairnessView, 1);


    // 信号与槽
    connect(openAct, SIGNAL(triggered()), this, SLOT(openFile()));
    connect(saveAct, SIGNAL(triggered()), this, SLOT(saveFile()));
    connect(saveasAct, SIGNAL(triggered()), this, SLOT(saveasFile()));
    connect(aboutQtAct, SIGNAL(triggered()), qApp, SLOT(aboutQt()));

    QMetaObject::connectSlotsByName(this);
}

/* 打开一个运行时输出信息txt文件 */
void MainWindow::openFile(){
    if(this->ui->tabWidget->currentIndex() == 0){  // ROBUST
        this->robustnessView->openFile();
    }else{
        this->fairnessView->openFile();
    }
}

/* 将运行时输出信息存为txt文件 */
void MainWindow::saveFile()
{
    if(this->ui->tabWidget->currentIndex() == 0){  // ROBUST
        this->robustnessView->saveFile();
    }else{
        this->fairnessView->saveFile();
    }
}

void MainWindow::saveasFile()
{
    if(this->ui->tabWidget->currentIndex() == 0){  // ROBUST
        this->robustnessView->saveasFile();
    }else{
        this->fairnessView->saveasFile();
    }
}

MainWindow::~MainWindow()
{
    delete ui;
}

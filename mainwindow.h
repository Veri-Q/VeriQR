#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QtCore/QVariant>
#include <QMenu>
#include <QMenuBar>
#include <QWidget>
#include <QFile>
#include <QFileInfo>
#include <QProcess>

#include "ui_mainwindow.h"
#include "localView.h"
#include "globalView.h"


QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE


class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    void init_GUI();


private:
    Ui::MainWindow *ui;

    LocalView *localView;
    GlobalView *globalView;

    QProcess* process;
    QMenu *fileMenu, *helpMenu;  //菜单栏
    QAction *openAct;
    QAction *saveAct;
    QAction *saveasAct;
    QAction *aboutQtAct;


private slots:
    void openFile();
    void saveFile();
    void saveasFile();
};
#endif // MAINWINDOW_H

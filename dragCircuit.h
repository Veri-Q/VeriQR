#ifndef DRAGCIRCUIT_H
#define DRAGCIRCUIT_H

#include <iostream>

#include <QWidget>
#include "ui_dragcircuit.h"

namespace Ui {
class dragCircuit;
}

class DragCircuit : public QWidget
{
    Q_OBJECT

public:
    explicit DragCircuit(QWidget *parent = nullptr);
    ~DragCircuit();

    void init();
    void show_all_gates();
    void mousePressEvent(QMouseEvent *event);
    void dragEnterEvent(QDragEnterEvent *event);  // 拖动进入事件
    void dragMoveEvent(QDragMoveEvent *event);    // 拖动事件
    void dropEvent(QDropEvent *event);            // 拖放事件
    QPoint getNearestPointOfNet(QPoint p);


private:
    Ui::dragCircuit *ui;

    Qt::DropAction actionMoveOrCopy;

    int num_qubit = 4;

    QPoint line_startPoint;
    QPoint line_endPoint;

    int lineHeight;
    int disOfLine;

    std::map<QString, int> numOfGateMap;
};

#endif // DRAGCIRCUIT_H

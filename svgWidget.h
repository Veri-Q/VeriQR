#ifndef SVGWIDGET_H
#define SVGWIDGET_H

#include <QSvgWidget>
#include <QSvgRenderer>
#include <QWidget>
#include <QtSvg>

class SvgWidget : public QSvgWidget
{
public:
    SvgWidget(QWidget *parent = nullptr);

    void wheelEvent(QWheelEvent *event);
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);

private:
    QSvgRenderer *svgRender;
    QPoint lastPos;
    bool isDragging;
};

#endif // SVGWIDGET_H

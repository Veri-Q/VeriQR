#include "svgWidget.h"

SvgWidget::SvgWidget(QWidget *parent)
{
    svgRender = this->renderer();
    this->resize(svgRender->defaultSize()); // Display the SVG file at its default size.
}

/* Respond to mouse scroll events, so that SVG images can be zoomed by mouse scrolling. */
void SvgWidget::wheelEvent(QWheelEvent *e)
{
    const double diff = 0.2; // 表示每次滚轮滚动一定的值, 图片大小改变的比例

    QSize size = svgRender->defaultSize();  //用于获取图片显示区的尺寸
    int width = size.width();
    int height = size.height();

    // 利用QWheelEvent的相关函数获得滚轮滚动的距离值, 通过此值判断滚轮滚动的方向
    // 若此值大于0, 则表示滚轮向前(远离用户方向)滚动;
    // 若此值小于0,则表示滚轮向后(靠近用户方向)滚动;
    if(e->angleDelta().y() > 0){
        // 对图片的长、宽值进行放大
        width = int(this->width() + this->width()*diff);
        height = int(this->height() + this->height()*diff);
    }else{
        // 对图片的长、宽值进行缩小
        width = int(this->width() - this->width()*diff);
        height = int(this->height() - this->height()*diff);
    }
    this->resize(width, height); // 重新调整大小
}

void SvgWidget::mousePressEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton)
    {
        lastPos = event->pos();
        setCursor(Qt::ClosedHandCursor);
        isDragging = true;
    }
}

void SvgWidget::mouseMoveEvent(QMouseEvent *event)
{
    if (isDragging)
    {
        int dx = event->position().x() - lastPos.x();
        int dy = event->position().y() - lastPos.y();
        lastPos = event->pos();
        svgRender->setViewBox(svgRender->viewBox().translated(-dx, -dy));
        update();
    }
}

void SvgWidget::mouseReleaseEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton)
    {
        setCursor(Qt::OpenHandCursor);
        isDragging = false;
    }
}

#ifndef PDFVIEW_H
#define PDFVIEW_H

#include <QtPdf>
#include <QtPdfWidgets>

class PdfView : public QPdfView
{
public:
    PdfView(QWidget *parent = nullptr);

    void loadDocument(QString filename);

    void wheelEvent(QWheelEvent *event); // 响应鼠标的滚动事件，使SVG图片可以通过鼠标滚动进行缩放

private:
    QPdfDocument *doc;
};

#endif // PDFVIEW_H

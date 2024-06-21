#include "pdfView.h"

PdfView::PdfView(QWidget *parent)
{
    this->setZoomFactor(2.5);
    // this->setZoomMode(QPdfView::ZoomMode::FitInView); // 适合页面宽度
    this->setZoomMode(QPdfView::ZoomMode::FitToWidth); // 适合窗口宽度
    // this->setPageMode(QPdfView::PageMode::MultiPage); // 连续页面显示
    // qDebug() << this->zoomFactor();
}

void PdfView::loadDocument(QString filename)
{
    doc = new QPdfDocument();
    doc->load(filename);

    if (!doc)  // 处理错误，例如文件无法打开
    {
        delete doc;
    }

    // const auto documentTitle = doc->metaData(QPdfDocument::MetaDataField::Title).toString();
    // QString title = !documentTitle.isEmpty() ? documentTitle : QStringLiteral("PDF doc");

    this->setDocument(doc);
}

void PdfView::wheelEvent(QWheelEvent *e)
{
    const double diff = 0.1; // 表示每次滚轮滚动一定的值, 图片大小改变的比例

    if(e->angleDelta().y() > 0){
        // 对图片的长、宽值进行放大
        // this->setZoomMode(QPdfView::ZoomMode::Custom);
        this->setZoomFactor(this->zoomFactor()* (1+diff));
    }else{
        // 对图片的长、宽值进行缩小
        // this->setZoomMode(QPdfView::ZoomMode::Custom);
        this->setZoomFactor(this->zoomFactor()* (1-diff));
    }
    // qDebug() << this->zoomFactor();

}

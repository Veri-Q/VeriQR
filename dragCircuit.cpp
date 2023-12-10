#include "dragCircuit.h"
#include <math.h>
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <QFileDialog>
#include <QTextStream>
#include <QDebug>
#include <QtCore5Compat/QTextCodec>
#include <QImage>
#include <QPainter>
#include <QPaintEvent>
#include <QMimeData>
#include <QDrag>

DragCircuit::DragCircuit(QWidget *parent):
    QWidget(parent),
    ui(new Ui::dragCircuit)
{
    ui->setupUi(this);

    this->setMouseTracking(true);
    this->setAcceptDrops(true);

    this->init();
}

void DragCircuit::init()
{
    line_startPoint = ui->line_q0->pos();
    line_endPoint = ui->line_q3->pos();

    lineHeight = ui->line_q0->height();
    disOfLine = ui->widget_q1->pos().y() - ui->widget_q0->pos().y();

    show_all_gates();
}

void DragCircuit::show_all_gates()
{
    QString img_file = ":/gates/images";

    QDir dir(img_file);
    QStringList mImgNames;

    if (!dir.exists()) mImgNames = QStringList("");

    dir.setFilter(QDir::Files);
    dir.setSorting(QDir::Time);  // 按创建时间排序遍历

    mImgNames = dir.entryList();
    qDebug() << dir.entryList();
    for (int i = mImgNames.size()-1; i >= 0 ; i--)
    {
        QString name = mImgNames[i];
        // qDebug() << "entryList: " << i << " " << name;

        QLabel *imageLabel_gate;
        imageLabel_gate= new QLabel();
        imageLabel_gate->setObjectName(QString::fromUtf8("imageLabel_gate_")
                                       + name.mid(name.lastIndexOf("/")+1, name.indexOf(".")));
        imageLabel_gate->setMouseTracking(true);

        QImage image(img_file + "/" + name);
        QPixmap pixmap = QPixmap::fromImage(image);
        imageLabel_gate->setPixmap(pixmap.scaledToHeight(ui->scrollAreaWidgetContents->height()*2, Qt::SmoothTransformation));

        ui->horizontalLayout_2->addWidget(imageLabel_gate);

        numOfGateMap[imageLabel_gate->objectName()] = 0; // 初始化为0
    }

    QImage image(":/gates/images/add_qubit.png");
    QPixmap pixmap = QPixmap::fromImage(image);
    ui->imageLabel_addqubit->setPixmap(pixmap);
}

void DragCircuit::mousePressEvent(QMouseEvent *event)
{
    QPoint posMouse = event->pos();

    if(event->button() != Qt::LeftButton)
        return;

    QWidget *label = static_cast<QWidget*>(this->childAt(posMouse));

    QString childName = label->objectName();
    qDebug() << childName;

    /* 单击了控件内部任何位置 */
    if(childName.contains("imageLabel_gate"))
    {
        QLabel *label_gate = static_cast<QLabel*>(label);
        QPixmap image = label_gate->pixmap();

        QWidget *childTmp;  // 这个变量的目的是不影响child的值
        childTmp = this->findChild<QLabel *>(childName)->parentWidget();
        // change to label's parent widget, because need move widget

        // 第二步：自定义MIME类型
        QByteArray itemData;                                     // 创建字节数组
        QDataStream dataStream(&itemData, QIODevice::WriteOnly); // 创建数据流: 将图片信息, 位置信息输入到字节数组中

        QPoint widgetPos = childTmp->pos();  // 拖动元件的widget坐标    /// ****这个地方有问题！！！
        QPoint offset = posMouse - label_gate->pos();      // 鼠标位置和widget的左上角位置坐标差
        qDebug() << offset << " = " << posMouse << " - " << label_gate->pos();
        dataStream << widgetPos << offset << childName;  // 注：这里只能先输入坐标数据，因为后面也是按照顺序拿的数据

        actionMoveOrCopy = Qt::CopyAction;
        //        dataStream << this->getIdWidget(childName);

        // 第三步：将数据放入QMimeData中
        QMimeData *mimeData = new QMimeData;  // 创建QMimeData用来存放要移动的数据
        mimeData->setData("myimage/png", itemData); // 将字节数组放入QMimeData中，这里的MIME类型是我们自己定义的

        // 第四步：将QMimeData数据放入QDrag中
        QDrag *drag = new QDrag(this);      // 创建QDrag，它用来移动数据
        drag->setMimeData(mimeData);
        drag->setPixmap(image);
        drag->setHotSpot(event->pos() - label_gate->pos());
        // drag->setHotSpot(offset); // 拖动时鼠标指针的位置不变

        // 第五步：执行拖放操作   这里我把它改为复制操作了
        if (drag->exec(Qt::CopyAction | Qt::MoveAction, Qt::MoveAction) == Qt::MoveAction)
        // 设置拖放可以是移动和复制操作，默认是复制操作
        {
            qDebug() << "other!";
            // child->close();   // 如果是移动操作，那么拖放完成后关闭原标签
        }
        else
        {
            qDebug() << "copy!";
            label_gate->show();   // 如果是复制操作，那么拖放完成后显示标签
        }
    }
    else if(childName.contains("imageLabel_addqubit"))
    {
        // add new qubit
        QString index = QString::number(num_qubit);

        QFont font;
        font.setPointSize(12);
        font.setBold(true);
        if(label->inherits("QLabel"))
        {
            QLabel *cur_label = qobject_cast<QLabel*>(label);
            cur_label->setObjectName(QString::fromUtf8("label_q") + index);
            cur_label->setText(QString::fromUtf8("q") + index);
            cur_label->setFont(font);
            cur_label->setAlignment(Qt::AlignCenter);
            cur_label->setCursor(QCursor(Qt::ArrowCursor));
        }

        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);

        // add new line
        QWidget *widget_q = new QWidget(ui->scrollAreaWidgetContents_circ_2);
        widget_q->setObjectName(QString::fromUtf8("widget_q") + index);
        sizePolicy.setHeightForWidth(widget_q->sizePolicy().hasHeightForWidth());
        widget_q->setSizePolicy(sizePolicy);

        QHBoxLayout *horizontalLayout = new QHBoxLayout(widget_q);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout_q") + index);

        QLabel *label = new QLabel(widget_q);
        label->setObjectName(QString::fromUtf8("imageLabel_addqubit"));

        QImage image(":/gates/images/add_qubit.png");
        QPixmap pixmap = QPixmap::fromImage(image);
        label->setPixmap(pixmap);
        label->setAlignment(Qt::AlignCenter);
        label->setCursor(QCursor(Qt::PointingHandCursor));
        horizontalLayout->addWidget(label);

        QFrame *line = new QFrame(widget_q);
        line->setObjectName(QString::fromUtf8("line_q") + index);
        line->setFrameShape(QFrame::HLine);
        line->setFrameShadow(QFrame::Sunken);

        horizontalLayout->addWidget(line);

        horizontalLayout->setStretch(0, 1);
        horizontalLayout->setStretch(1, 15);

        ui->verticalLayout_7->addWidget(widget_q);

        qDebug() << ui->scrollAreaWidgetContents_circ_2->children();
        num_qubit++;
    }
}

void DragCircuit::dragEnterEvent(QDragEnterEvent *event)
{
    qDebug() << "This is a dragEnterEvent";
    if (event->mimeData()->hasFormat("myimage/png"))
    {
        event->setDropAction(actionMoveOrCopy);   // 如果有我们定义的MIME类型数据，则进行移动操作
        event->accept();
    }
    else
    {
        event->ignore();
    }
}

void DragCircuit::dragMoveEvent(QDragMoveEvent *event)
{
    //    qDebug() << "This is a dragEnterEvent";
    if (event->mimeData()->hasFormat("myimage/png")) {
        event->setDropAction(Qt::MoveAction);
        event->accept();
    }
    else {
        event->ignore();
    }
}

void DragCircuit::dropEvent(QDropEvent *event)
{
       if (event->mimeData()->hasFormat("myimage/png"))
       {
           QByteArray itemData = event->mimeData()->data("myimage/png");
           QDataStream dataStream(&itemData, QIODevice::ReadOnly);
           QPoint offset, parentWidgetPos; //这是被拖元件原来的底widget的坐标
           QString childName;
           dataStream >> parentWidgetPos >> offset >> childName;  // 使用数据流将字节数组中的数据读入到QPixmap和QPoint变量中
           qDebug() << "childName: " << childName;

           if(!childName.contains("imageLabel_gate"))
           {
               event->ignore();
           }

           QLabel *imagelabel_gate;
           imagelabel_gate = this->findChild<QLabel *>(childName);
           qDebug() << imagelabel_gate;
    //        qDebug() << child->objectName();

           // QPoint currentPos = event->pos() - offset;
           QPoint currentPos = event->pos();
           // qDebug() << currentPos << " = " << event->pos() << " - " << offset;

           // currentPos = getNearestPointOfNet(currentPos); // 只能停靠在Line上
           if(currentPos == parentWidgetPos)
           {
               qDebug() << "Position unchanged, no drag! ";
               return;
           }

           QWidget *nearWidget = static_cast<QWidget*>(this->childAt(currentPos));
           QString cur_qubitName = nearWidget->objectName().mid(nearWidget->objectName().indexOf("_")+1);
           qDebug() << nearWidget->objectName();
           qDebug() << cur_qubitName;

           QLabel *new_imageLabel = new QLabel(nearWidget);
           new_imageLabel->setObjectName(childName + "_" + cur_qubitName + "_" + QString::number(++numOfGateMap[childName]));
           new_imageLabel->move(event->pos());
           new_imageLabel->show();
           qDebug() << new_imageLabel->objectName();

           QPixmap pixmap = imagelabel_gate->pixmap();
           new_imageLabel->setPixmap(pixmap);
           new_imageLabel->setGeometry(QRect(currentPos.x(), currentPos.y(),
                                             imagelabel_gate->width(), imagelabel_gate->height()));

           update();
           nearWidget->layout()->addWidget(new_imageLabel);

           qDebug() << actionMoveOrCopy;
           event->setDropAction(actionMoveOrCopy);
           event->accept();
       }
       else
       {
           event->ignore();
       }
}

QPoint DragCircuit::getNearestPointOfNet(QPoint p)
{
    //    QPoint setLinePoint;
    qDebug() << "p.y(): " << p.y();
    qDebug() << "line_startPoint.y(): " << line_startPoint.y();
    qDebug() << "disOfLine: " << disOfLine;

    int numberOfline = (p.y() - line_startPoint.y()) / disOfLine;
    int nearest_line_y = line_startPoint.y() + numberOfline * disOfLine;  // 计算最邻近鼠标位置的上一条线的y
    qDebug() << "nearest_line_y: " << nearest_line_y;

    QPoint res;
    int d = p.y() - (nearest_line_y + lineHeight);
    if(d <= (disOfLine-lineHeight)/2)  // 如果鼠标的y更加靠近上面一条直线的y，就选这条直线的y坐标作为停靠坐标
    {
        res.setY(nearest_line_y);
        //        setLinePoint = new QPoint(line_startPoint.x(), nearest_line_y);
    }
    else                               // 鼠标的y更加靠近下面一条直线的y
    {
        res.setY(nearest_line_y + disOfLine);
        //        setLinePoint = new QPoint(line_startPoint.x(),  + disOfLine);
    }

    //    QWidget *setLine = static_cast<QWidget*>(this->childAt(setLinePoint));

    // 现在设置x停靠
    //    numberOfline = (p.x() - line_startPoint.x()) / disOfLine;
    //    nearest_line_y = line_startPoint.x() + numberOfline * disOfLine; // 得到鼠标y坐标的上面一条线的那个y
    //    d = p.x() - nearest_line_y;
    //    if(d <= disOfLine/2)
    //        res.setX(nearest_line_y); // 如果鼠标的y更加靠近上面一条直线的y，那就选这条直线的y坐标作为停靠坐标
    //    else
    //        res.setX(line_startPoint.x() + (numberOfline+1)*disOfLine); // 如果鼠标的y更加靠近下面一条直线的y，那就选这条直线的y坐标作为停靠坐标
    //    QPoint res(p.x(), setLinePoint.y());
    res.setX(p.x());

    return res;
}

DragCircuit::~DragCircuit()
{
    delete ui;
}

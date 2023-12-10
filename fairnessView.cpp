#include "fairnessView.h"
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <dirent.h>
#include <math.h>
#include <regex>

#include <QMessageBox>
#include <QFileDialog>
#include <QTextStream>
#include <QDebug>
#include <QImage>
#include <QPainter>
#include <QPaintEvent>


FairnessView::FairnessView(QWidget *parent):
    QWidget(parent),
    ui(new Ui::fairnessView)
{
    ui->setupUi(this);
    this->init();

    connect(ui->radioButton_importfile, SIGNAL(pressed()), this, SLOT(on_radioButton_importfile_clicked()));
    connect(ui->radioButton_gc, SIGNAL(pressed()), this, SLOT(on_radioButton_gc_clicked()));
    connect(ui->radioButton_dice, SIGNAL(pressed()), this, SLOT(on_radioButton_dice_clicked()));
    connect(ui->radioButton_phaseflip, SIGNAL(toggled(bool)), this, SLOT(on_radioButton_phaseflip_clicked()));
    connect(ui->radioButton_bitflip, SIGNAL(toggled(bool)), this, SLOT(on_radioButton_bitflip_clicked()));
    connect(ui->radioButton_depolarize, SIGNAL(toggled(bool)), this, SLOT(on_radioButton_depolarize_clicked()));
    connect(ui->radioButton_mixed, SIGNAL(toggled(bool)), this, SLOT(on_radioButton_mixed_clicked()));
    connect(ui->slider_prob, SIGNAL(valueChanged(int)), this, SLOT(on_slider_prob_sliderMoved(int)));
    connect(ui->doubleSpinBox_prob, SIGNAL(valueChanged(double)), this, SLOT(on_doubleSpinBox_prob_valueChanged(double)));
    connect(ui->pushButton_run, SIGNAL(pressed()), this, SLOT(run_fairnessVeri()));
    connect(ui->pushButton_stop, SIGNAL(pressed()), this, SLOT(stopProcess()));
 }

void FairnessView::init()
{
    QString path = QApplication::applicationFilePath();
    if(path.contains("/build-VeriQR"))
    {
        fairDir = path.mid(0, path.indexOf("/build-VeriQR")) + "/VeriQR/py_module/Fairness";
    }
    else if(path.contains("VeriQR/build/"))
    {
        fairDir = path.mid(0, path.indexOf("/build")) + "/py_module/Fairness";
    }
    qDebug() << "fairDir: " << fairDir;
}

void FairnessView::resizeEvent(QResizeEvent *)
{
    if(showed_loss)
    {
        show_loss_and_acc_plot();
    }
    // if(showed_svg)
    // {
    //     show_circuit_diagram();
    // }
}

bool FairnessView::findModel(QString filename)
{
    QString filePath = fairDir + "/saved_model/" + filename;
    qDebug() << filePath;

    struct stat s;

    if (stat(filePath.toStdString().c_str(), &s) == 0)
    {
        if (s.st_mode & S_IFDIR || s.st_mode & S_IFREG)
        {
            qDebug() << "This is a directory or file.";
        }
        else
        {
            qDebug() << "This is not a directory or file.";
        }
        return true;
    }

    qDebug() << "The file don't exist! ";
    return false;
}

void FairnessView::clear_all_information()
{
    output_ = "";
    output_line_ = "";
    ui->lineEdit_k->setText("");
    ui->lineEdit_time->setText("");
    ui->textBrowser_output->setText("");
    ui->imageLabel_plot->clear();
    showed_loss = false;

    if(showed_svg){
        delete_circuit_diagram();
    }
}

/* 打开一个运行时输出信息txt文件 */
void FairnessView::openFile(){
    QString fileName = QFileDialog::getOpenFileName(this, "Open file", fairDir+"/training_output/");
    QFile file(fileName);

    if (file.open(QIODevice::ReadOnly |QIODevice::Text) && QFileInfo(fileName).suffix() == "txt")
    {
        // clear all
        clear_all_information();

        // 从文件名获取各种参数信息
        file_name_ = QFileInfo(file).fileName();  // gc_phase_flip_0.0001.txt
        file_name_.chop(4);
        qDebug() << file_name_;

        QStringList args = file_name_.split("_");
        noise_prob_ = args[args.size()-1].toDouble();

        if(args.size() == 4){  // "phase_flip" or "bit_flip"
            noise_type_ = args[1] + "_" + args[2];
        }
        else if(args.size() == 3){
            noise_type_ = args[1];
        }

        // UI change
        model_change_to_ui();

        while (!file.atEnd())
        {
            output_line_ = QString(file.readLine());
            output_.append(output_line_);
            ui->textBrowser_output->append(output_line_.simplified());

            if(output_line_.contains("Training End"))
            {
                show_loss_and_acc_plot();
            }
            else if(output_line_.contains("Printing Model Circuit End"))
            {
                show_circuit_diagram();
            }
            else if(output_line_.startsWith("Lipschitz K"))
            {
                lipschitz_ = output_line_.mid(output_line_.indexOf("=  ")+3).toDouble();
                ui->lineEdit_k->setText(QString::number(lipschitz_));
                qDebug() << output_line_;
                qDebug() << lipschitz_;
            }
            else if(output_line_.startsWith("Elapsed time"))
            {
                int a = output_line_.indexOf("=") + 2;
                veri_time_ = output_line_.mid(a, output_line_.size()-a-2).toDouble();
                ui->lineEdit_time->setText(QString::number(veri_time_)+"s");
                qDebug() << output_line_;
                qDebug() << veri_time_;
            }
            else if(output_.contains("Lipschitz Constant End")){
                break;
            }
        }
    }
    else
    {
        QMessageBox::warning(this, "Warning", "Unable to open the file: " + file.errorString());
        return;
    }

    file.close();
}

/* 将运行时输出信息存为txt文件 */
void FairnessView::saveFile()
{
    if(ui->textBrowser_output->toPlainText().isEmpty()){
        QMessageBox::warning(this, "Warning", "No program was ever run and no results can be saved.");
        return;
    }

    output_ = ui->textBrowser_output->toPlainText();

    QString runtime_path = fairDir + "/training_output/" + file_name_ + ".txt";
    qDebug() << runtime_path;

    QFile file(runtime_path);
    if(file.open(QIODevice::WriteOnly| QIODevice::Text| QIODevice::Truncate)){
        QTextStream stream(&file);
        stream << output_;
        file.close();
        QMessageBox::information(NULL, "",
                                 "The file " + runtime_path + " was saved successfully!",
                                 QMessageBox::Yes);
    }
}

void FairnessView::saveasFile()
{
    QString fileName = QFileDialog::getSaveFileName(this, "Save as", fairDir + "/training_output/");
    QFile file(fileName);

    if (!file.open(QFile::WriteOnly | QFile::Text)) {
        QMessageBox::warning(this, "Warning", "Unable to open the file: " + file.errorString());
        return;
    }
    QTextStream out(&file);
    out << output_;
    QMessageBox::information(NULL, "",
                             "The file " + file.fileName() + " was saved successfully!",
                             QMessageBox::Yes);
    file.close();
}

void FairnessView::on_radioButton_gc_clicked(){
    pyfile_ = pyfiles[0];
}

void FairnessView::on_radioButton_dice_clicked(){
    pyfile_ = pyfiles[1];
}

void FairnessView::on_radioButton_importfile_clicked(){
    if(ui->radioButton_importfile->isChecked()){
        importModel();
    }
    pyfile_ = "";
}

/* 导入.npz数据文件 */
void FairnessView::importModel()
{
    //    QString fileName = QFileDialog::getOpenFileName(this, "Open file", fairDir);
    //    QFile file(fileName);
    //    current_fileinfo_ = QFileInfo(fileName);

    //    if (!file.open(QIODevice::ReadOnly)) {
    //        QMessageBox::warning(this, "Warning", "Unable to open the file: " + file.errorString());
    //        return;
    //    }else if(current_fileinfo_.suffix() != "npz"){
    //        QMessageBox::warning(this, "Warning", "VeriQFair only supports .npz data files.");
    //        return;
    //    }
    //    ui->lineEdit_filepathname->setText(current_fileinfo_.filePath());

    //    file.close();
}

void FairnessView::on_radioButton_phaseflip_clicked()
{
    noise_type_ = noise_types[0];
    qDebug() << noise_type_;
}

void FairnessView::on_radioButton_bitflip_clicked()
{
    noise_type_ = noise_types[1];
    qDebug() << noise_type_;
}

void FairnessView::on_radioButton_depolarize_clicked()
{
    noise_type_ = noise_types[2];
    qDebug() << noise_type_;
}

void FairnessView::on_radioButton_mixed_clicked()
{
    noise_type_ = noise_types[3];
    qDebug() << noise_type_;
}

void FairnessView::model_change_to_ui(){
    // selected model change
    if(file_name_.startsWith("gc")){
        ui->radioButton_gc->setChecked(1);
    }else if(file_name_.startsWith("dice")){
        ui->radioButton_dice->setChecked(1);
    }else{
        ui->radioButton_importfile->setChecked(1);
    }

    // noise type change
    if(noise_type_ == "phase_flip"){
        ui->radioButton_phaseflip->setChecked(1);
    }
    else if(noise_type_ == "bit_flip"){
        ui->radioButton_bitflip->setChecked(1);
    }
    else if(noise_type_ == "depolarize"){
        ui->radioButton_depolarize->setChecked(1);
    }
    else if(noise_type_ == "mixed"){
        ui->radioButton_mixed->setChecked(1);
    }

    // noise probability change
    ui->doubleSpinBox_prob->setValue(noise_prob_);
    qDebug() << ui->slider_prob->value();
}

void FairnessView::run_fairnessVeri()
{
    QString choice = "train";

    noise_prob_ = ui->doubleSpinBox_prob->value();

    clear_all_information();

    if (!pyfile_.isEmpty()){  // has selected a existing model file
        QString model_name = pyfile_.mid(pyfile_.lastIndexOf("_") + 1); // 去掉evaluate_finance_model_前缀

        QStringList list;
        list << model_name << noise_type_ << QString::number(noise_prob_);
        file_name_ = list.join("_");   // csv和txt结果文件的默认命名
        qDebug() << file_name_;

        if(findModel(file_name_))  // the current model has been trained and saved before
        {
            QString dlgTitle = "Select an option";
            QString strInfo = "The model has been trained and saved before. "
                              "Do you want to review the existing model or retrain it?";
            QMessageBox msgBox;
            msgBox.setWindowTitle(dlgTitle);
            msgBox.setText(strInfo);
            QPushButton *showButton = msgBox.addButton(tr("Show the existing model"), QMessageBox::ActionRole);
            QPushButton *trainButton = msgBox.addButton(tr("Retrain"),QMessageBox::ActionRole);
            //            QPushButton *cancelButton = msgBox.addButton(QMessageBox::Cancel);
            msgBox.addButton(QMessageBox::No);
            msgBox.button(QMessageBox::No)->setHidden(true);
            msgBox.setDefaultButton(QMessageBox::NoButton);
            msgBox.exec();

            if (msgBox.clickedButton() == showButton)
            {
                choice = "notrain";
            }
        }

        QString cmd = "python3";
        QStringList args;
        args << pyfile_+".py" << noise_type_ << QString::number(noise_prob_) << choice;

        QString paramsList = pyfile_ + ".py " + noise_type_ + " " + QString::number(noise_prob_) + " " + choice;
        qDebug() << paramsList;

        process = new QProcess(this);
        process->setReadChannel(QProcess::StandardOutput);
        connect(process, SIGNAL(stateChanged(QProcess::ProcessState)), SLOT(stateChanged(QProcess::ProcessState)));
        connect(process, SIGNAL(readyReadStandardOutput()), this, SLOT(on_read_from_terminal()));
        //        connect(process, SIGNAL(finished(int, QProcess::ExitStatus)), this, SLOT(show_results(int, QProcess::ExitStatus)));

        process->setWorkingDirectory(fairDir);
        process->start(cmd, args);
        if(!process->waitForStarted()){
            qDebug() << "Process failure! Error: " << process->errorString();
        }
        else{
            qDebug() << "Process succeed! ";
        }

        if (!process->waitForFinished()) {
            qDebug() << "wait";
            QCoreApplication::processEvents(QEventLoop::AllEvents, 2000);
        }

        QString error = process->readAllStandardError(); // 命令行执行出错的提示
        if(!error.isEmpty()){
            qDebug()<< "Error executing script: " << error; // 打印出错提示
        }

    }else{
        // TODO

    }
}

void FairnessView::stateChanged(QProcess::ProcessState state)
{
    qDebug()<<"show state:";
    switch(state)
    {
    case QProcess::NotRunning:
        qDebug()<<"Not Running";
        break;
    case QProcess::Starting:
        qDebug()<<"Starting";
        break;
    case QProcess::Running:
        qDebug()<<"Running";
        break;
    default:
        qDebug()<<"otherState";
        break;
    }
}

void FairnessView::on_read_from_terminal()
{
    while (process->bytesAvailable() > 0){
        output_line_ = process->readLine();
        output_.append(output_line_);
        ui->textBrowser_output->append(output_line_.simplified());
        //        qDebug() << output_line_;

        if(output_line_.contains("Training End"))
        {
            show_loss_and_acc_plot();
        }
        else if(output_line_.contains("Printing Model Circuit End"))
        {
            show_circuit_diagram();
        }
        else if(output_line_.startsWith("Lipschitz K"))
        {
            lipschitz_ = output_line_.mid(output_line_.indexOf("K =  ")+5).toDouble();
            ui->lineEdit_k->setText(QString::number(lipschitz_));
            qDebug() << output_line_;
            qDebug() << lipschitz_;
        }
        else if(output_line_.startsWith("Elapsed time"))
        {
            int a = output_line_.indexOf("= ") + 2;
            veri_time_ = output_line_.mid(a, output_line_.size()-a-2).toDouble();
            ui->lineEdit_time->setText(QString::number(veri_time_)+"s");
            qDebug() << output_line_;
            qDebug() << veri_time_;
        }
        else if(output_.contains("Lipschitz Constant End")){
            break;
        }
    }
}

void FairnessView::show_loss_and_acc_plot()
{
    QString img_file = fairDir + "/result_figures/" + file_name_ + ".png";
    qDebug() << img_file;

    QImage image(img_file);
    QPixmap pixmap = QPixmap::fromImage(image);
    ui->imageLabel_plot->setPixmap(pixmap.scaled(ui->tabWidget->width()*0.8, ui->tabWidget->height()*0.8,
                                                 Qt::KeepAspectRatio, Qt::SmoothTransformation));
    showed_loss = true;
}

void FairnessView::show_circuit_diagram()
{
    QString img_file = fairDir + "/model_circuits/circuit_" + file_name_ + ".svg";
    qDebug() << "img_file: " << img_file;

    svgWidget = new SvgWidget(ui->scrollArea_circ);
    svgWidget->load(img_file);
    svgWidget->setObjectName("svgWidget_circ");

    ui->verticalLayout_circ->insertWidget(0, svgWidget, 1);
    ui->verticalLayout_circ->setStretch(1, 3);

    // svgRender = new QSvgRenderer();
    // svgRender->load(img_file);
    // QSize size = svgRender->defaultSize(); // 获取svg的大小
    // QPixmap pixmap(size * 1.8);            // 在这给绘图设备重新设置大小
    // //        QPixmap pixmap = QPixmap(1024,1024);
    // pixmap.fill(Qt::transparent); // 设置背景透明, 像素清空, 这一步必须有, 否则背景有黑框
    // QPainter painter(&pixmap);
    // painter.setRenderHints(QPainter::Antialiasing|QPainter::TextAntialiasing| QPainter::SmoothPixmapTransform);  // 反锯齿绘制
    // svgRender->render(&painter);
    // ui->imageLabel_circ->setPixmap(pixmap.scaledToHeight(ui->scrollAreaWidgetContents->height()*0.8,
    //                                                      Qt::SmoothTransformation));
    showed_svg = true;
}

void FairnessView::delete_circuit_diagram()
{
    QWidget *svgwidget = ui->verticalLayout_circ->itemAt(0)->widget();
    svgwidget->setParent (NULL);
    qDebug() << "delete " << svgwidget->objectName() << "!";

    this->ui->verticalLayout_circ->removeWidget(svgwidget);
    delete svgwidget;

    showed_svg = false;
}

void FairnessView::stopProcess()
{
    this->process->terminate();
    this->process->waitForFinished();

    QMessageBox::information(this, "Notice", "The program was terminated.");
    qDebug() << "Process terminate!";
}

FairnessView::~FairnessView()
{
    delete ui;
}

void FairnessView::on_slider_prob_sliderMoved(int pos)
{
    ui->doubleSpinBox_prob->setValue(ui->slider_prob->value()* 0.001);
    noise_prob_ = ui->doubleSpinBox_prob->value();
    qDebug() << noise_prob_;
}

void FairnessView::on_doubleSpinBox_prob_valueChanged(double pos)
{
    ui->slider_prob->setValue(ui->doubleSpinBox_prob->value()/ 0.001);
    noise_prob_ = ui->doubleSpinBox_prob->value();
    qDebug() << noise_prob_;
}

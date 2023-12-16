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
}

bool FairnessView::findFile(QString filename)
{
    struct stat s;

    if (stat(filename.toStdString().c_str(), &s) == 0)
    {
        if (s.st_mode & S_IFDIR || s.st_mode & S_IFREG)
        {
            qDebug() << "This is a directory or file.";
        }
        else
        {
            qDebug() << "This is not a directory or file.";
        }
        qDebug() << "has find the file: " << filename;
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
    QString fileName = QFileDialog::getOpenFileName(this, "Open file", fairDir+"/output/");

    show_saved_results(fileName);
}

void FairnessView::show_saved_results(QString fileName)
{
    // clear all
    clear_all_information();

    // 从文件名获取各种参数信息
    QFile file(fileName);
    qDebug() << QFileInfo(file).filePath();
    if (!file.open(QIODevice::ReadOnly |QIODevice::Text) || QFileInfo(fileName).suffix() != "txt")
    {
        QMessageBox::warning(this, "Warning", "Unable to open the file: " + file.errorString());
        return;
    }

    file_name_ = QFileInfo(file).fileName();  // gc_phase_flip_0.0001.txt  or  hf_6_0_5_bit_flip_0.01.txt
    file_name_.chop(4);
    qDebug() << "file_name_: " << file_name_;

    QStringList args = file_name_.split("_");
    noise_prob_ = args[args.size()-1].toDouble();

    if(args[args.size()-2] == "flip"){  // "phase_flip" or "bit_flip"
        noise_type_ = args[args.size()-3] + "_" + args[args.size()-2];
    }
    else{
        noise_type_ = args[args.size()-2];
    }
    qDebug() << noise_prob_;
    qDebug() << noise_type_;

    // UI change
    model_change_to_ui();

    while (!file.atEnd())
    {
        output_line_ = QString(file.readLine());
        output_.append(output_line_);
        ui->textBrowser_output->append(output_line_.simplified());
        // qDebug() << output_line_;

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
            lipschitz_ = output_line_.mid(output_line_.indexOf("= ")+2).toDouble();
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
        else if(output_line_.contains("The Lipschitz Constant Calculation End")){
            break;
        }
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

    QString runtime_path = fairDir + "/output/" + file_name_ + ".txt";
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
    QString fileName = QFileDialog::getSaveFileName(this, "Save as", fairDir + "/output/");
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
    model_file_.fileName().clear();
    ui->lineEdit_modelfile->clear();
}

void FairnessView::on_radioButton_dice_clicked(){
    pyfile_ = pyfiles[1];
    model_file_.fileName().clear();
    ui->lineEdit_modelfile->clear();
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
   QString fileName = QFileDialog::getOpenFileName(this, "Open file", fairDir+"/qasm_models");
   QFile file(fileName);
   model_file_ = QFileInfo(fileName);

   if (!file.open(QIODevice::ReadOnly)) {
       QMessageBox::warning(this, "Warning", "Unable to open the file: " + file.errorString());
       return;
   }else if(model_file_.suffix() != "qasm"){
       QMessageBox::warning(this, "Warning", "VeriQR only supports .qasm model files.");
       return;
   }
   ui->lineEdit_modelfile->setText(model_file_.filePath());

   file.close();
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
    clear_all_information();

    noise_prob_ = ui->doubleSpinBox_prob->value();

    if (!pyfile_.isEmpty())  // has selected a existing model file
    {
        QString model_name = pyfile_.mid(pyfile_.lastIndexOf("_") + 1); // 去掉evaluate_finance_model_前缀
        QString choice = "train";

        QStringList list;
        list << model_name << noise_type_ << QString::number(noise_prob_);
        file_name_ = list.join("_");   // like: gc_phase_flip_0.0001
        qDebug() << "file_name_: " << file_name_;

        if(findFile(fairDir + "/saved_models/" + file_name_))  // the current model has been trained and saved before
        {
            QString dlgTitle = "Select an option";
            QString strInfo = "The model has been trained and saved before. "
                              "Do you want to review the previous model or retrain it?";
            QMessageBox msgBox;
            msgBox.setWindowTitle(dlgTitle);
            msgBox.setText(strInfo);
            QPushButton *showButton = msgBox.addButton(tr("Show the previous model"), QMessageBox::ActionRole);
            QPushButton *trainButton = msgBox.addButton(tr("Retrain"),QMessageBox::ActionRole);
            msgBox.addButton(QMessageBox::No);
            msgBox.button(QMessageBox::No)->setHidden(true);
            msgBox.setDefaultButton(QMessageBox::NoButton);
            msgBox.exec();

            if (msgBox.clickedButton() == showButton)
            {
                choice = "notrain";
            }
        }

        QString cmd = "python";
        QStringList args;
        args << pyfile_+".py" << noise_type_ << QString::number(noise_prob_) << choice;

        QString paramsList = cmd + " " + args.join(" ");
        qDebug() << paramsList;

        exec_process(cmd, args);
    }
    else  // has selected another .qasm file
    {
        QString model_name = model_file_.fileName();  // like: hf_6_0_5.qasm
        model_name.chop(5);   // 去掉.qasm
        qDebug() << "model_name: " << model_name;

        QStringList list;
        list << model_name << noise_type_ << QString::number(noise_prob_);
        file_name_ = list.join("_");   // like: hf_6_0_5_bit_flip_0.01
        qDebug() << "file_name_: " << file_name_;

        // python qlipschitz.py qasmfile phase_flip 0.0001
        QString cmd = "python";
        QStringList args;
        args << "qlipschitz.py" << model_file_.filePath() << noise_type_ << QString::number(noise_prob_);
        QString paramsList = cmd + " " + args.join(" ");
        qDebug() << paramsList;

        QString strInfo = "The results of this model have been saved before. "
                          "Do you want to see the previous results or recalculate it?";
        QMessageBox msgBox;
        msgBox.setWindowTitle("Select an option");
        msgBox.setText(strInfo);
        QPushButton *showButton = msgBox.addButton(tr("Show the previous results"), QMessageBox::ActionRole);
        QPushButton *calcButton = msgBox.addButton(tr("Recalculate"),QMessageBox::ActionRole);
        msgBox.addButton(QMessageBox::No);
        msgBox.button(QMessageBox::No)->setHidden(true);
        msgBox.setDefaultButton(QMessageBox::NoButton);

        QString file = fairDir + "/output/" + file_name_ + ".txt";

        // if(findFile(file) && (msgBox.exec() && msgBox.clickedButton() == showButton))  // the current model has been trained and saved before
        // {

        //     show_saved_results(file);
        // }
        // else
        // {
        //     exec_process(cmd, args);
        // }

        if(findFile(file))  // the current model has been trained and saved before
        {
            msgBox.exec();
            if(msgBox.clickedButton() == showButton){
                show_saved_results(file);
            }
            else{
                exec_process(cmd, args);
            }
        }
        else{
            exec_process(cmd, args);
        }
    }
}

void FairnessView::exec_process(QString cmd, QStringList args)
{
    process = new QProcess(this);
    process->setReadChannel(QProcess::StandardOutput);
    connect(process, SIGNAL(stateChanged(QProcess::ProcessState)), SLOT(stateChanged(QProcess::ProcessState)));
    connect(process, SIGNAL(readyReadStandardOutput()), this, SLOT(on_read_from_terminal()));

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
            lipschitz_ = output_line_.mid(output_line_.indexOf("= ")+2).toDouble();
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
        else if(output_line_.contains("The Lipschitz Constant Calculation End")){
            break;
        }
    }
}

void FairnessView::show_loss_and_acc_plot()
{
    QString img_file = fairDir + "/loss_figures/" + file_name_ + ".png";
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

    svgWidget = new SvgWidget(ui->scrollAreaWidgetContents_circ);
    svgWidget->load(img_file);
    svgWidget->setObjectName("svgWidget_circ");

    double container_w = double(ui->scrollAreaWidgetContents_circ->width());
    double svg_w = double(svgWidget->renderer()->defaultSize().width());
    double svg_h = double(svgWidget->renderer()->defaultSize().height());
    double iris_w = 977.0;
    double iris_h = 260.0;
    svg_h = container_w / svg_w * svg_h;
    iris_h = container_w / iris_w * iris_h;
    // qDebug() << svg_h;
    // iris.svg: (width, height) = (977, 260)

    QSpacerItem *verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Expanding, QSizePolicy::Expanding);
    ui->verticalLayout_circ->addItem(verticalSpacer);
    int diff = double(svg_h*2)/double(iris_h) * 1000;
    // qDebug() << diff;
    ui->verticalLayout_circ->insertWidget(0, svgWidget, diff);
    ui->verticalLayout_circ->setStretch(1, 3*1000);

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

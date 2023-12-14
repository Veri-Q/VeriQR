#include "robustnessView.h"
#include <math.h>
#include <iostream>
#include <fstream>
#include <sys/stat.h>

#include <QMessageBox>
#include <QFileDialog>
#include <QTextStream>
#include <QIcon>
#include <QFont>
#include <QFontDialog>
#include <QDebug>
#include <QtCore5Compat/QTextCodec>
#include <QImage>
#include <QPainter>
#include <QPaintEvent>
#include <QMimeData>
#include <QDrag>
#include <QtConcurrent/QtConcurrent>
#include <QFuture>


RobustnessView::RobustnessView(QWidget *parent):
    QWidget(parent),
    ui(new Ui::robustnessView)
{
    qputenv("QTWEBENGINE_REMOTE_DEBUGGING", "7777");
    ui->setupUi(this);
    this->init();

    connect(ui->radioButton_importfile, SIGNAL(pressed()), this, SLOT(on_radioButton_importfile_clicked()));
    connect(ui->pushButton_importdata, SIGNAL(pressed()), this, SLOT(importData()));
    connect(ui->radioButton_binary, SIGNAL(pressed()), this, SLOT(on_radioButton_binary_clicked()));
    connect(ui->radioButton_phaseRecog, SIGNAL(pressed()), this, SLOT(on_radioButton_phaseRecog_clicked()));
    connect(ui->radioButton_excitation, SIGNAL(pressed()), this, SLOT(on_radioButton_excitation_clicked()));
    connect(ui->radioButton_mnist, SIGNAL(pressed()), this, SLOT(on_radioButton_mnist_clicked()));
    connect(ui->radioButton_pure, SIGNAL(toggled(bool)), this, SLOT(on_radioButton_pure_clicked()));
    connect(ui->radioButton_mixed, SIGNAL(toggled(bool)), this, SLOT(on_radioButton_mixed_clicked()));
    connect(ui->checkBox, SIGNAL(stateChanged(int)), this, SLOT(on_checkBox_clicked(int)));
    connect(ui->slider_unit, SIGNAL(valueChanged(int)), this, SLOT(on_slider_unit_sliderMoved(int)));
    connect(ui->spinBox_unit, SIGNAL(valueChanged(int)), this, SLOT(on_spinBox_unit_valueChanged(int)));
    connect(ui->slider_exptnum, SIGNAL(valueChanged(int)), this, SLOT(on_slider_exptnum_sliderMoved(int)));
    connect(ui->spinBox_exptnum, SIGNAL(valueChanged(int)), this, SLOT(on_spinBox_exptnum_valueChanged(int)));
    connect(ui->pushButton_run, SIGNAL(pressed()), this, SLOT(run_robustVeri()));
    connect(ui->pushButton_stop, SIGNAL(pressed()), this, SLOT(stopProcess()));
}

void RobustnessView::init()
{
    QString path = QApplication::applicationFilePath();
    if(path.contains("/build-VeriQR"))
    {
        robustDir = path.mid(0, path.indexOf("/build-VeriQR")) + "/VeriQR/py_module/Robustness";
    }
    else if(path.contains("VeriQR/build/"))
    {
        robustDir = path.mid(0, path.indexOf("/build")) + "/py_module/Robustness";
    }
    qDebug() << path;
    qDebug() << "robustDir: " << robustDir;

    ui->textBrowser_output->setGeometry(QRect(20, 10, 67, 17));

    QPalette palette = ui->groupBox_run->palette();
    palette.setBrush(QPalette::Window, Qt::transparent);  // 设置背景颜色为透明
    ui->groupBox_run->setPalette(palette);

    palette = ui->groupBox_runtime->palette();
    palette.setBrush(QPalette::Window, Qt::transparent);
    ui->groupBox_runtime->setPalette(palette);

    // dragCircuit = new DragCircuit(ui->tab_drag_circ);
    // dragCircuit->setObjectName(QString::fromUtf8("dragCircuit"));
    // ui->verticalLayout_7->addWidget(dragCircuit);

    // webView = new QWebEngineView(ui->scrollAreaWidgetContents_2);
    // palette = webView->palette();
    // palette.setBrush(QPalette::Base, Qt::transparent);  // 设置背景颜色为透明
    // webView->setPalette(palette);
    // setAttribute(Qt::WA_OpaquePaintEvent,false);

    // webView->load(QUrl("https://hiq.huaweicloud.com/portal/programming/hiq-composer?id=UntitledCircuit&type=circuit"));
    // webView->show();

    // ui->horizontalLayout_6->addWidget(webView);
}

void RobustnessView::resizeEvent(QResizeEvent *)
{
    if(showed_svg)
    {
        svgWidget->load(robustDir + "/Figures/"+ model_name_ + "_model.svg");
    }

    if(showed_adexample)
    {
        QString img_file = robustDir + "/adversary_examples";
        QDir dir(img_file);
        QStringList mImgNames;

        if (!dir.exists()) mImgNames = QStringList("");

        dir.setFilter(QDir::Files);
        dir.setSorting(QDir::Name);
        mImgNames = dir.entryList();

        int i = 0;
        QObjectList list = ui->scrollAreaWidgetContents_ad->children();
        foreach(QObject *obj, list)
        {
            if(obj->inherits("QLabel"))
            {
                QLabel *imageLabel = qobject_cast<QLabel*>(obj);
                QImage image(img_file + "/" + mImgNames[i]);
                QPixmap pixmap = QPixmap::fromImage(image);
                imageLabel->setPixmap(pixmap.scaledToHeight(
                    ui->tab_ad->height()*0.32, Qt::SmoothTransformation));
                // qDebug() << imageLabel->objectName();
                i++;
            }
        }
    }
}

/* 打开一个运行时输出信息txt文件 */
void RobustnessView::openFile(){
    QString fileName = QFileDialog::getOpenFileName(this, "Open file", robustDir+"/results/runtime_output");
    QFile file(fileName);

    if (!file.open(QIODevice::ReadOnly |QIODevice::Text)) {
        QMessageBox::warning(this, "Warning", "Unable to open the file: " + file.errorString());
        return;
    }else if(QFileInfo(fileName).suffix() != "txt") {
        QMessageBox::warning(this, "Warning", "VeriQRobust only supports .txt result data files.");
        return;
    }

    close_circuit_diagram();
    delete_all_adversary_examples();

    output_ = QString::fromLocal8Bit(file.readAll());
    ui->textBrowser_output->setText(output_);

    file_name_ = QFileInfo(file).fileName();
    file_name_.chop(4);
    qDebug() << file_name_;  // "binary_0.001_2_mixed"

    csvfile_ = robustDir + "/results/result_tables/" + file_name_ + ".csv";
    // "../RobustnessVerifier/py_module/Robustness/results/result_tables/binary_0.001_2_mixed.csv"

    // 此时需要从文件名获取各种参数信息
    QStringList args = file_name_.split("_");
    model_name_ = args[0];
    robustness_unit_ = args[1].toDouble();
    experiment_number_ = args[2].toInt();
    state_type_ = args[3];

    ui->slider_unit->setValue(args[1].length() - 2);
    ui->spinBox_unit->setValue(args[1].length() - 2);

    ui->slider_exptnum->setValue(experiment_number_);
    ui->spinBox_exptnum->setValue(experiment_number_);

    if(model_name_ == "binary"){
        ui->radioButton_binary->setChecked(1);
    }
    else if(model_name_ == "phaseRecog"){
        ui->radioButton_phaseRecog->setChecked(1);
    }
    else if(model_name_ == "excitation"){
        ui->radioButton_excitation->setChecked(1);
    }
    else if(model_name_ == "mnist"){
        ui->radioButton_mnist->setChecked(1);
    }
    else{
        ui->radioButton_importfile->setChecked(1);
    }

    if(state_type_ == "pure"){
        ui->radioButton_pure->setChecked(1);
    }
    else if(state_type_ == "mixed"){
        ui->radioButton_mixed->setChecked(1);
    }

    if(model_name_ == "mnist" && state_type_ == "pure"){
        ui->checkBox->setChecked(1);
        show_adversary_examples();
    }

    show_result_tables();

    // get_table_data(1, QProcess::NormalExit);
    get_table_data("openfile");

    show_circuit_diagram();

    file.close();
}

/* 将运行时输出信息存为txt文件 */
void RobustnessView::saveFile()
{
    if(ui->textBrowser_output->toPlainText().isEmpty()){
        QMessageBox::warning(this, "Warning", "No program was ever run and no results can be saved.");
        return;
    }

    output_ = ui->textBrowser_output->toPlainText();

    QString runtime_path = robustDir + "/results/runtime_output/" + file_name_ + ".txt";
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

void RobustnessView::saveasFile()
{
    QString fileName = QFileDialog::getSaveFileName(this, "Save as", robustDir + "/results/runtime_output/");
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

/* 导入.qasm数据文件 */
void RobustnessView::importModel()
{
    QString fileName = QFileDialog::getOpenFileName(this, "Open file", robustDir+"/model_and_data");
    QFile file(fileName);
    model_file_ = QFileInfo(fileName);

    model_name_ = fileName.mid(fileName.lastIndexOf("/")+1, fileName.indexOf(".")-fileName.lastIndexOf("/")-1);
    qDebug() << "model_name_: " << model_name_;

    if (!file.open(QIODevice::ReadOnly)) {
        QMessageBox::warning(this, "Warning", "Unable to open the file: " + file.errorString());
        return;
    }else if(model_file_.suffix() != "qasm"){
        QMessageBox::warning(this, "Warning", "VeriQRobust only supports .npz data files.");
        return;
    }

    ui->lineEdit_modelfile->setText(model_file_.filePath());

    file.close();
}

void RobustnessView::on_radioButton_importfile_clicked(){
    if(ui->radioButton_importfile->isChecked()){
        importModel();
    }
    npzfile_ = "";
}

/* 导入.npz数据文件 */
void RobustnessView::importData(){
    if(!ui->radioButton_importfile->isChecked()){
        QMessageBox::warning(this, "Warning", "Please select the model first! ");
        return;
    }

    QString fileName = QFileDialog::getOpenFileName(this, "Open file", robustDir+"/model_and_data");
    QFile file(fileName);
    data_file_ = QFileInfo(fileName);

    if (!file.open(QIODevice::ReadOnly)) {
        QMessageBox::warning(this, "Warning", "Unable to open the file: " + file.errorString());
        return;
    }else if(data_file_.suffix() != "npz"){
        QMessageBox::warning(this, "Warning", "VeriQRobust only supports .npz data files.");
        return;
    }

    ui->lineEdit_datafile->setText(data_file_.filePath());

    file.close();
}

void RobustnessView::on_radioButton_binary_clicked()
{
    npzfile_ = "binary_cav.npz";
    model_name_ = npzfile_.mid(0, npzfile_.indexOf("_"));
    qDebug() << npzfile_;

    model_file_.fileName().clear();
    data_file_.fileName().clear();
    ui->lineEdit_modelfile->clear();
    ui->lineEdit_datafile->clear();

    if(ui->checkBox->isChecked()){
        ui->checkBox->setChecked(0);  // 取消选中
    }
}

void RobustnessView::on_radioButton_phaseRecog_clicked()
{
    npzfile_ = "phaseRecog_cav.npz";
    model_name_ = npzfile_.mid(0, npzfile_.indexOf("_"));
    qDebug() << npzfile_;

    model_file_.fileName().clear();
    data_file_.fileName().clear();
    ui->lineEdit_modelfile->clear();
    ui->lineEdit_datafile->clear();

    if(ui->checkBox->isChecked()){
        ui->checkBox->setChecked(0);  // 取消选中
    }
}

void RobustnessView::on_radioButton_excitation_clicked()
{
    npzfile_ = "excitation_cav.npz";
    model_name_ = npzfile_.mid(0, npzfile_.indexOf("_"));
    qDebug() << npzfile_;

    model_file_.fileName().clear();
    data_file_.fileName().clear();
    ui->lineEdit_modelfile->clear();
    ui->lineEdit_datafile->clear();

    if(ui->checkBox->isChecked()){
        ui->checkBox->setChecked(0);  // 取消选中
    }
}

void RobustnessView::on_radioButton_mnist_clicked()
{
    npzfile_ = "mnist_cav.npz";
    model_name_ = npzfile_.mid(0, npzfile_.indexOf("_"));
    qDebug() << npzfile_;

    model_file_.fileName().clear();
    data_file_.fileName().clear();
    ui->lineEdit_modelfile->clear();
    ui->lineEdit_datafile->clear();
}

void RobustnessView::on_radioButton_pure_clicked()
{
    state_type_ = "pure";
    qDebug() << state_type_;
}

void RobustnessView::on_radioButton_mixed_clicked()
{
    state_type_ = "mixed";
    qDebug() << state_type_;

    if(ui->checkBox->isChecked()){
        ui->checkBox->setChecked(0);  // 取消选中
    }
}

void RobustnessView::on_slider_unit_sliderMoved(int pos)
{
    ui->spinBox_unit->setValue(pos);
    //    qDebug() << robustness_unit_;
}

void RobustnessView::on_slider_exptnum_sliderMoved(int pos)
{
    experiment_number_ = pos;
    ui->spinBox_exptnum->setValue(pos);
}

void RobustnessView::on_spinBox_unit_valueChanged(int pos)
{
    ui->slider_unit->setValue(pos);
    //    qDebug() << robustness_unit_;
}

void RobustnessView::on_spinBox_exptnum_valueChanged(int pos)
{
    experiment_number_ = pos;
    ui->slider_exptnum->setValue(pos);
}

void RobustnessView::run_robustVeri()
{
    if((model_file_.fileName().isEmpty() || data_file_.fileName().isEmpty())  // 未选择任何模型
        && npzfile_ == "")
    {
        QMessageBox::warning(this, "Warning", "You should choose a model! ");
        return;
    }

    output_ = "";
    output_line_ = "";
    ui->textBrowser_output->setText("");

    close_circuit_diagram();
    delete_all_adversary_examples();

    robustness_unit_ = pow(0.1, ui->slider_unit->value());
    experiment_number_ = ui->slider_exptnum->value();

    QString excuteFile = "batch_check.py";
    QString unit = QString::number(robustness_unit_);
    QString exptnum = QString::number(experiment_number_);
    QString paramsList;
    QStringList args;
    qDebug() << exptnum;

    if(model_file_.fileName().isEmpty() || data_file_.fileName().isEmpty())
    {
        QString npzfile = "./model_and_data/" + npzfile_; // npzfile is complete path
        paramsList = excuteFile + " " + npzfile + " " + unit + " " + exptnum + " " + state_type_;
        args << excuteFile << npzfile << unit << exptnum << state_type_;
    }
    else    // has imported a file
    {
        QString qasmfile = model_file_.filePath();
        QString datafile = data_file_.filePath();
        paramsList = excuteFile + " " + qasmfile + " " + datafile + " " + unit + " " + exptnum + " " + state_type_;
        args << excuteFile << qasmfile << datafile << unit << exptnum << state_type_;
    }

    // model_name is fileName without "_cav.npz" suffix like "binary"
    if(model_name_ == "mnist")
    {
        if(state_type_ == "pure"){
            ui->checkBox->setChecked(1);
            paramsList += " true";
            args << "true";
        }else{
            ui->checkBox->setChecked(0);
            paramsList += " false";
            args << "false";
        }
    }
    qDebug() << paramsList;

    QStringList list;
    list << model_name_ << unit << exptnum << state_type_;
    file_name_ = list.join("_");   // csv和txt结果文件的默认命名
    qDebug() << file_name_;

    csvfile_ = robustDir + "/results/result_tables/" + file_name_ + ".csv";

    show_result_tables();

    QString cmd = "python3";

    process = new QProcess(this);
    process->setReadChannel(QProcess::StandardOutput);
    connect(process, SIGNAL(stateChanged(QProcess::ProcessState)), SLOT(stateChanged(QProcess::ProcessState)));
    connect(process, SIGNAL(readyReadStandardOutput()), this, SLOT(on_read_output()));
    // connect(process, SIGNAL(finished(int, QProcess::ExitStatus)), this, SLOT(get_table_data(int, QProcess::ExitStatus)));
    process->setWorkingDirectory(robustDir);
    process->start(cmd, args);
    if(!process->waitForStarted())
    {
        qDebug() << "Process failure! Error: " << process->errorString();
    }else
    {
        qDebug() << "Process succeed! ";
    }

    if (!process->waitForFinished()) {
        qDebug() << "wait";
        QCoreApplication::processEvents(QEventLoop::AllEvents, 2000);
    }

    QString error =  process->readAllStandardError(); //命令行执行出错的提示
    if(!error.isEmpty()){
        qDebug()<< "Error executing script： " << error; //打印出错提示
    }
}

void RobustnessView::stateChanged(QProcess::ProcessState state)
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

void RobustnessView::stopProcess()
{
    this->process->terminate();
    this->process->waitForFinished();
    qDebug() << "Process terminate!";
}

void RobustnessView::on_read_output()
{
    while (process->bytesAvailable() > 0){
        output_line_ = process->readLine();
        // qDebug() << output_line_;

        if(output_line_.contains("Starting") && !showed_pdf && !showed_svg)
        {
            show_circuit_diagram();
        }
        // Verification over, show adversary examples
        else if(output_line_.contains("Robust Accuracy (in Percent)"))
        {
            if(ui->checkBox->isChecked()){
                show_adversary_examples();
            }
            break;
        }
        output_.append(output_line_);
        ui->textBrowser_output->append(output_line_.simplified());
    }
    get_table_data("run");
}

void RobustnessView::show_adversary_examples()
{
    QString img_file = robustDir + "/adversary_examples";
    // qDebug() << img_file;

    QDir dir(img_file);
    QStringList mImgNames;

    if (!dir.exists()) mImgNames = QStringList("");

    dir.setFilter(QDir::Files);
    dir.setSorting(QDir::Name);

    mImgNames = dir.entryList();
    qDebug() << dir.entryList();
    for (int i = 0; i < mImgNames.size(); ++i)
    {
        QLabel *imageLabel_ad;
        imageLabel_ad= new QLabel(ui->scrollAreaWidgetContents_ad);
        imageLabel_ad->setObjectName(QString::fromUtf8("imageLabel_ad_")+QString::number(i));
        imageLabel_ad->setAlignment(Qt::AlignCenter);

        QImage image(img_file + "/" + mImgNames[i]);
        QPixmap pixmap = QPixmap::fromImage(image);
        imageLabel_ad->setPixmap(pixmap);

        ui->verticalLayout_4->addWidget(imageLabel_ad);
    }
    showed_adexample = true;
}

void RobustnessView::delete_all_adversary_examples()
{
    if(!showed_adexample)  // 当前没有展示adversary examples, 就不需要做任何处理
        return;

    QLayoutItem *child;
    int i=0;
    while((child = ui->verticalLayout_4->takeAt(i)) != nullptr)
    {
        if(child->widget())
        {
            child->widget()->setParent(nullptr);
            qDebug() << "delete " << child->widget()->objectName() << "!";
            ui->verticalLayout_4->removeWidget(child->widget());
            delete child->widget();
        }
    }
    showed_adexample = false;
    qDebug() << "delete all adversary examples!";
}

void RobustnessView::close_circuit_diagram()
{
    if(showed_svg){
        close_circuit_diagram_svg();
        showed_svg = false;
    }
    else if(showed_pdf){
        close_circuit_diagram_pdf();
        showed_pdf = false;
    }
}

void RobustnessView::close_circuit_diagram_svg()
{
    QLayoutItem *child;
    int i=0;
    while((child = ui->verticalLayout_circ->takeAt(i)) !=nullptr)
    {
        if(child->widget())
        {
            child->widget()->setParent(nullptr);
            qDebug() << "delete " << child->widget()->objectName() << "!";
            ui->verticalLayout_circ->removeWidget(child->widget());
            delete child->widget();
        }
    }
}

void RobustnessView::close_circuit_diagram_pdf()
{
    QLayoutItem *child = ui->verticalLayout_circ->takeAt(0);
    if(child !=nullptr)
    {
        if(child->widget())
        {
            child->widget()->setParent(nullptr);
            qDebug() << "delete " << child->widget()->objectName() << "!";
            pdfView->document()->close();
            ui->verticalLayout_circ->removeWidget(child->widget());
            delete child->widget();
        }
    }
}

void RobustnessView::show_circuit_diagram()
{
    QString img_file = robustDir + "/Figures/";

    if(circuit_diagram_map.find(model_name_) != circuit_diagram_map.end())
    {
        img_file += circuit_diagram_map[model_name_];
        qDebug() << "img_file: " << img_file;

        show_circuit_diagram_pdf(img_file);
        showed_pdf = true;
    }
    else
    {
        img_file += model_name_ + "_model.svg";
        qDebug() << "img_file: " << img_file;

        show_circuit_diagram_svg(img_file);
        showed_svg = true;
    }
}

void RobustnessView::show_circuit_diagram_svg(QString filename)
{
    svgWidget = new SvgWidget(ui->scrollAreaWidgetContents_circ);
    svgWidget->load(filename);
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
}

void RobustnessView::show_circuit_diagram_pdf(QString filename)
{
    pdfView = new PdfView(ui->scrollArea_circ);
    pdfView->loadDocument(filename);
    pdfView->setObjectName("pdfView_circ");
    ui->verticalLayout_circ->addWidget(pdfView);
}

void RobustnessView::show_result_tables(){
    int columnCount = experiment_number_;
    accuracy_model = new QStandardItemModel();
    accuracy_model->setColumnCount(columnCount);

    times_model = new QStandardItemModel();
    times_model->setColumnCount(columnCount);

    // 设置列表头
    int eps = ui->slider_unit->value();
    for(int i = 1; i <= columnCount; ++i){
        QString str = QString::number(i)+QString::fromLocal8Bit("e-")+QString::number(eps);
        accuracy_model->setHeaderData(i-1, Qt::Horizontal, str);
        times_model->setHeaderData(i-1, Qt::Horizontal, str);
    }

    // 设置行表头
    accuracy_model->setVerticalHeaderLabels(QStringList() << "Robust Bound" << "Robustness Algorithm");
    times_model->setVerticalHeaderLabels(QStringList() << "Robust Bound" << "Robustness Algorithm");

    item = new QStandardItem(QString("Robust Bound"));
    //    item->setData(QColor(Qt::gray), Qt::BackgroundRole);
    //    item->setData(QColor(Qt::white), Qt::FontRole);
    item->setTextAlignment(Qt::AlignCenter);
    accuracy_model->setVerticalHeaderItem(0, item);
    item = new QStandardItem(QString("Robust Bound"));
    item->setTextAlignment(Qt::AlignCenter);
    times_model->setVerticalHeaderItem(0, item);

    item = new QStandardItem(QString("Robustness Algorithm"));
    item->setTextAlignment(Qt::AlignCenter);
    accuracy_model->setVerticalHeaderItem(1, item);
    item = new QStandardItem(QString("Robust Algorithm"));
    item->setTextAlignment(Qt::AlignCenter);
    times_model->setVerticalHeaderItem(1, item);

    //在QTableView中加入模型
    ui->table_accuracy->setModel(accuracy_model);
    ui->table_times->setModel(times_model);

    QString  header_style = "QHeaderView::section{"
                           "background:rgb(120,120,120);"
                           "color:rgb(255,255,255);"
                           "padding: 1px;}";

    ui->table_accuracy->setColumnWidth(0, 80);
    ui->table_accuracy->setColumnWidth(1, 80);
    ui->table_accuracy->setShowGrid(true);
    ui->table_accuracy->setGridStyle(Qt::DotLine);
    ui->table_accuracy->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    ui->table_accuracy->horizontalHeader()->setStyleSheet(header_style);
    ui->table_accuracy->verticalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    ui->table_accuracy->verticalHeader()->setStyleSheet(header_style);
    //    ui->table_accuracy->verticalHeader()->setDefaultSectionSize(100);
    //    ui->table_accuracy->verticalHeader()->sectionResizeMode(QHeaderView::Stretch);
    //    ui->table_accuracy->verticalHeader()->sectionResizeMode(QHeaderView::ResizeToContents);

    ui->table_times->setColumnWidth(0,80);
    ui->table_times->setColumnWidth(1,110);
    ui->table_times->setShowGrid(true);
    ui->table_times->setGridStyle(Qt::DotLine);
    ui->table_times->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    ui->table_times->horizontalHeader()->setStyleSheet(header_style);
    ui->table_times->verticalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    ui->table_times->verticalHeader()->setStyleSheet(header_style);

    for(int i = 0; i < 2; ++i){
        for(int j = 0; j < columnCount; ++j){
            item = new QStandardItem(QString("-"));
            item->setTextAlignment(Qt::AlignCenter);
            accuracy_model->setItem(i, j, item);
            item = new QStandardItem(QString("-"));
            item->setTextAlignment(Qt::AlignCenter);
            times_model->setItem(i, j, item);
        }
    }
}

/* 将文件内容解析到表格 */
void RobustnessView::get_table_data(QString op){
    // 程序异常结束
    if(op == "run")
    {
        qDebug() << process->exitStatus();
        if(process->exitStatus() != QProcess::NormalExit){
            QMessageBox::warning(this, "Warning", "Program abort.");
            return;
        }
    }

    // 程序正常结束
    QFile file(csvfile_);
    qDebug() << csvfile_;

    if(!file.open(QIODevice::ReadOnly | QIODevice::Text)){
        QMessageBox::warning(this, "Warning", "Unable to open the .csv file: " + csvfile_ + "\n" + file.errorString());
        return;
    }

    QTextStream in(&file);
    int row = 1;
    while (!in.atEnd()) {
        QString line = in.readLine();
        if(row == 2){
            QStringList ac_fields = line.split(",");
            //            qDebug() << ac_fields[1] << " " << ac_fields[2];
            for(int j = 0; j < experiment_number_; ++j){
                item = new QStandardItem(QString(ac_fields[j+1]));
                item->setTextAlignment(Qt::AlignCenter);
                accuracy_model->setItem(0, j, item);
                //                accuracy_model->item(0, j)->setText(ac_fields_1[j+1]);
            }
        }
        if(row == 3){
            QStringList ac_fields = line.split(",");
            for(int j = 0; j < experiment_number_; ++j){
                item = new QStandardItem(QString(ac_fields[j+1]));
                item->setTextAlignment(Qt::AlignCenter);
                accuracy_model->setItem(1, j, item);
            }
        }
        if(row == 6){
            QStringList time_fields = line.split(",");
            for(int j = 0; j < experiment_number_; ++j){
                item = new QStandardItem(QString(time_fields[j+1]));
                item->setTextAlignment(Qt::AlignCenter);
                times_model->setItem(0, j, item);
            }
        }
        if(row == 7){
            QStringList time_fields = line.split(",");
            for(int j = 0; j < experiment_number_; ++j){
                item = new QStandardItem(QString(time_fields[j+1]));
                item->setTextAlignment(Qt::AlignCenter);
                times_model->setItem(1, j, item);
            }
        }
        row++;
    }
    file.close();
}

void RobustnessView::on_checkBox_clicked(int state)
{
    if(state == Qt::Checked || state == Qt::PartiallyChecked){
        if(model_name_ == "mnist" && state_type_ == "pure"){
            ui->checkBox->setChecked(1);
        }
        else
        {
            ui->checkBox->setChecked(0);
            QMessageBox::warning(this, "Warning", "Only the mnist model in pure state supports generating adversary examples");
        }
    }
    qDebug() << ui->checkBox->isChecked();
}

RobustnessView::~RobustnessView()
{
    delete ui;
}

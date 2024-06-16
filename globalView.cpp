#include "globalView.h"
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


GlobalView::GlobalView(QWidget *parent):
    QWidget(parent),
    ui(new Ui::globalView)
{
    ui->setupUi(this);
    this->init();

    connect(ui->radioButton_importfile, SIGNAL(pressed()), this, SLOT(on_radioButton_importfile_clicked()));
    connect(ui->radioButton_cr, SIGNAL(pressed()), this, SLOT(on_radioButton_cr_clicked()));
    connect(ui->radioButton_aci, SIGNAL(pressed()), this, SLOT(on_radioButton_aci_clicked()));
    connect(ui->radioButton_fct, SIGNAL(pressed()), this, SLOT(on_radioButton_aci_clicked()));
    connect(ui->radioButton_phaseflip, SIGNAL(toggled(bool)), this, SLOT(on_radioButton_phaseflip_clicked()));
    connect(ui->radioButton_bitflip, SIGNAL(toggled(bool)), this, SLOT(on_radioButton_bitflip_clicked()));
    connect(ui->radioButton_depolarizing, SIGNAL(toggled(bool)), this, SLOT(on_radioButton_depolarizing_clicked()));
    connect(ui->radioButton_mixednoise, SIGNAL(toggled(bool)), this, SLOT(on_radioButton_mixednoise_clicked()));
    connect(ui->radioButton_custom_noise, SIGNAL(pressed()), this, SLOT(on_radioButton_importkraus_clicked()));
    connect(ui->slider_prob, SIGNAL(valueChanged(int)), this, SLOT(on_slider_prob_sliderMoved(int)));
    connect(ui->doubleSpinBox_prob, SIGNAL(valueChanged(double)), this, SLOT(on_doubleSpinBox_prob_valueChanged(double)));
    connect(ui->pushButton_run, SIGNAL(pressed()), this, SLOT(run_calculate_k()));
    connect(ui->pushButton_stop, SIGNAL(pressed()), this, SLOT(stopProcess()));
    connect(ui->slider_threshold_1, SIGNAL(valueChanged(int)), this, SLOT(on_slider_epsilon_sliderMoved(int)));
    connect(ui->doubleSpinBox_threshold_1, SIGNAL(valueChanged(double)), this, SLOT(on_doubleSpinBox_epsilon_valueChanged(double)));
    connect(ui->slider_threshold_2, SIGNAL(valueChanged(int)), this, SLOT(on_slider_delta_sliderMoved(int)));
    connect(ui->doubleSpinBox_threshold_2, SIGNAL(valueChanged(double)), this, SLOT(on_doubleSpinBox_delta_valueChanged(double)));
    connect(ui->pushButton_veri, SIGNAL(pressed()), this, SLOT(run_globalVeri()));
}

void GlobalView::init()
{
    QString path = QApplication::applicationFilePath();
    if(path.contains("/build-VeriQR"))
    {
        globalDir = path.mid(0, path.indexOf("/build-VeriQR")) + "/VeriQR/py_module/Global";
    }
    else if(path.contains("VeriQR/build/"))
    {
        globalDir = path.mid(0, path.indexOf("/build")) + "/py_module/Global";
    }
    qDebug() << "globalDir: " << globalDir;

    comboBox_mixednoise = new MultiSelectComboBox(ui->groupBox_noisetype);
    QSizePolicy sizePolicy1(QSizePolicy::Expanding, QSizePolicy::Fixed);
    sizePolicy1.setHorizontalStretch(0);
    sizePolicy1.setVerticalStretch(0);
    sizePolicy1.setHeightForWidth(comboBox_mixednoise->sizePolicy().hasHeightForWidth());
    comboBox_mixednoise->setSizePolicy(sizePolicy1);
    ui->gridLayout->addWidget(comboBox_mixednoise, 0, 5, 1, 1);
    QStringList noiseList;
    noiseList << "bit flip" << "depolarizing" << "phase flip";
    comboBox_mixednoise->setMaxSelectNum(3);
    comboBox_mixednoise->addItems_for_noise(noiseList);
}

void GlobalView::resizeEvent(QResizeEvent *)
{
    // if(showed_loss)
    // {
    //     show_loss_and_acc_plot();
    // }
}

bool GlobalView::findFile(QString filename)
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

void GlobalView::clear_output()
{
    output_line_.clear();
    output_.clear();
    ui->textBrowser_output->clear();

    ui->lineEdit_k->clear();
    ui->lineEdit_time->clear();

    close_circuit_diagram();
}

void GlobalView::reset_all()
{
    clear_output();

    // Reset the options in the interface and related variables.
    // Basic settings:
    model_file_ = QFileInfo();
    model_name_.clear();
    // filename_.clear();
    ui->radioButton_fct->setChecked(0);
    ui->radioButton_aci->setChecked(0);
    ui->radioButton_cr->setChecked(0);
    ui->radioButton_importfile->setChecked(0);
    ui->lineEdit_modelfile->clear();

    noise_type_.clear();
    ui->radioButton_bitflip->setChecked(0);
    ui->radioButton_depolarizing->setChecked(0);
    ui->radioButton_phaseflip->setChecked(0);
    ui->radioButton_mixednoise->setChecked(0);
    mixed_noises_.clear();
    comboBox_mixednoise->line_edit_->clear();
    for(int i = 0; i < comboBox_mixednoise->list_widget_->count(); i++)
    {
        QCheckBox *noise_check_box = static_cast<QCheckBox*>(
            comboBox_mixednoise->list_widget_->itemWidget(
                comboBox_mixednoise->list_widget_->item(i)
                )
            );
        noise_check_box->setChecked(0);
    }
    kraus_file_ = QFileInfo();
    ui->lineEdit_custom_noise->clear();
    ui->radioButton_custom_noise->setChecked(0);

    noise_prob_ = 0.0;
    ui->doubleSpinBox_prob->setValue(noise_prob_);
    ui->slider_prob->setValue(noise_prob_ / 0.00001);

    update();
    qDebug() << "Reset all settings! ";
}

/* Open a csv file that records verification results. */
void GlobalView::openFile(){
    QString fileName = QFileDialog::getOpenFileName(this, "Open file", globalDir+"/results/result_tables");
    QFile file(fileName);
    qDebug() << QFileInfo(file).filePath();

    if(!file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        QMessageBox::warning(this, "Warning",
                             QString("Unable to open the file: %1 \n").arg(fileName)
                                 + file.errorString());
        return;
    }
    else if(QFileInfo(fileName).suffix() != "csv") {
        QMessageBox::warning(this, "Warning",
                             QString("The suffix of the result data file %1 is not csv.")
                                 .arg(fileName));
        return;
    }

    // clear all
    reset_all();

    show_result_table();

    QTextStream in(&file);
    int row_index = 0;
    while (!in.atEnd()) {
        QString line = in.readLine();
        if(row_index > 0){
            QStringList res_fields = line.split(",");
            QString perturbations = QString("%1, %2").arg(QString(res_fields[1]),
                                                          QString(res_fields[2]));
            insert_data_to_table(row_index,
                                 QString(res_fields[0]),
                                 perturbations.remove('"'),
                                 QString(res_fields[3]),
                                 QString(res_fields[4]),
                                 QString(res_fields[5]));
        }
        row_index++;
    }
    file.close();
}

void GlobalView::show_result_table(){
    res_model = new QStandardItemModel();
    // Set the column header.
    res_model->setHorizontalHeaderLabels(QStringList() << "Circuit" << "Perturbations (ε, δ)" <<
                                                          "K*" << "VT(s)" << "If Robust");

    // Add QStandardItemModel to QTableView.
    ui->table_res->setModel(res_model);

    QString header_style = "QHeaderView::section{"
                           "background:rgb(120,120,120);"
                           "color:rgb(255,255,255);"
                           "padding: 1px;}";
    ui->table_res->setShowGrid(true);
    ui->table_res->setGridStyle(Qt::DotLine);
    // Remove the automatic serial number column.
    ui->table_res->verticalHeader()->setHidden(true);
    ui->table_res->verticalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    ui->table_res->verticalHeader()->setStyleSheet(header_style);
    ui->table_res->horizontalHeader()->setSectionResizeMode(QHeaderView::Interactive);
    ui->table_res->horizontalHeader()->setStyleSheet(header_style);
    ui->table_res->horizontalHeader()->setMinimumHeight(50);
    ui->table_res->setColumnWidth(0, ui->table_res->width()/8*3);
    ui->table_res->setColumnWidth(1, ui->table_res->width()/8*2);
    ui->table_res->setColumnWidth(2, ui->table_res->width()/8*1);
    ui->table_res->setColumnWidth(3, ui->table_res->width()/8*1);
    ui->table_res->setColumnWidth(4, ui->table_res->width()/8*1);
}

void GlobalView::insert_data_to_table(int row_index, QString circuit, QString perturbations,
                                      QString K, QString VT, QString robust)
{
    QStandardItem *circuit_item = new QStandardItem(circuit);
    QStandardItem *perturbation_item = new QStandardItem(perturbations);
    QStandardItem *K_item = new QStandardItem(K);
    QStandardItem *VT_item = new QStandardItem(VT);
    QStandardItem *robust_item = new QStandardItem(robust);
    circuit_item->setTextAlignment(Qt::AlignCenter);
    perturbation_item->setTextAlignment(Qt::AlignCenter);
    K_item->setTextAlignment(Qt::AlignCenter);
    VT_item->setTextAlignment(Qt::AlignCenter);
    robust_item->setTextAlignment(Qt::AlignCenter);
    res_model->setItem(row_index-1, 0, circuit_item);
    res_model->setItem(row_index-1, 1, perturbation_item);
    res_model->setItem(row_index-1, 2, K_item);
    res_model->setItem(row_index-1, 3, VT_item);
    res_model->setItem(row_index-1, 4, robust_item);
}

/* Open a txt file that records the runtime output. */
void GlobalView::show_saved_output(QString fileName)
{
    // QString fileName = QFileDialog::getOpenFileName(this, "Open file", globalDir+"/results/runtime_output");
    QFile file(fileName);

    if (!file.open(QIODevice::ReadOnly |QIODevice::Text)) {
        QMessageBox::warning(this, "Warning", "Unable to open the file: " + file.errorString());
        return;
    }
    else if(QFileInfo(fileName).suffix() != "txt") {
        QMessageBox::warning(this, "Warning", "VeriQR only supports .txt output information files.");
        return;
    }

    reset_all();

    output_ = QString::fromLocal8Bit(file.readAll());
    ui->textBrowser_output->setText(output_);
    file.close();

    file_name_ = QFileInfo(file).fileName();  // e.g. hf6_0.001_0.003_mixed_bitflip_0.13462.txt
    file_name_.chop(4);
    qDebug() << "file_name_: " << file_name_;

    // Obtain parameters from the filename
    QStringList args = file_name_.split("_");
    model_name_ = args[0];
    epsilon_ = args[1].toDouble();
    delta_ = args[2].toDouble();
    // model_name_ = file_name_.mid(0, file_name_.indexOf(args[1])-1);
    int noise_type_index = 3;
    noise_type_ = args[noise_type_index];
    noise_prob_ = args[args.size()-1].toDouble();
    ui->doubleSpinBox_prob->setValue(noise_prob_);

    // Noise settings change
    if(noise_type_ == "phaseflip"){
        ui->radioButton_phaseflip->setChecked(1);
        noise_type_ = "phase_flip";
    }
    else if(noise_type_ == "bitflip"){
        ui->radioButton_bitflip->setChecked(1);
        noise_type_ = "bit_flip";
    }
    else if(noise_type_ == "depolarizing"){
        ui->radioButton_depolarizing->setChecked(1);
        noise_type_ = "depolarizing";
        for(int i = noise_type_index + 1; i < args.size()-1; i++)
        {
            QString noise = noise_name_map[args[i]];
            mixed_noises_.append(noise);
        }
        comboBox_mixednoise->line_edit_->setText(mixed_noises_.join(";"));
    }
    else if(noise_type_ == "mixed"){
        ui->radioButton_mixednoise->setChecked(1);
    }
    else if(noise_type_ == "custom")  // hf6_0.001_0.003_custom_kraus1qubit_0.001
    {
        kraus_file_ = QFileInfo(globalDir + "/kraus/" + args[4] + ".npz");
        ui->lineEdit_custom_noise->setText(kraus_file_.filePath());
    }
    qDebug() << noise_prob_;
    qDebug() << noise_type_;

    // Selected model change
    if(file_name_.startsWith("cr")){
        ui->radioButton_cr->setChecked(1);
    }
    else if(file_name_.startsWith("aci")){
        ui->radioButton_aci->setChecked(1);
    }
    else if(file_name_.startsWith("fct")){
        ui->radioButton_fct->setChecked(1);
    }
    else{
        ui->radioButton_importfile->setChecked(1);
        // ui->lineEdit_modelfile->setText(globalDir+"/qasm_models/"+model_name_+".qasm");
    }

    while (!file.atEnd())
    {
        output_line_ = QString(file.readLine());
        output_.append(output_line_);
        ui->textBrowser_output->append(output_line_.simplified());
        // qDebug() << output_line_;

        if(output_line_.contains(".svg was saved"))
        {
            show_circuit_diagram(globalDir + "/figures/" +
                                 output_line_.mid(0, output_line_.indexOf(".svg was saved")+4));
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
            verif_time_ = output_line_.mid(a, output_line_.size()-a-2).toDouble();
            ui->lineEdit_time->setText(QString::number(verif_time_)+"s");
            qDebug() << output_line_;
            qDebug() << verif_time_;
        }
        else if(output_line_.contains("The Global Verification End")){
            break;
        }
    }

    file.close();
}

/* Save the runtime output as a txt file to the specified location. */
void GlobalView::saveFile()
{
    if(ui->textBrowser_output->toPlainText().isEmpty()){
        QMessageBox::warning(this, "Warning", "No program was ever run and no results can be saved.");
        return;
    }

    output_ = ui->textBrowser_output->toPlainText();

    QString runtime_path = globalDir + "/results/runtime_output/" + file_name_ + ".txt";
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

void GlobalView::saveasFile()
{
    QString fileName = QFileDialog::getSaveFileName(this, "Save as", globalDir + "/results/");
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

void GlobalView::on_radioButton_cr_clicked(){
    model_name_ = "cr";
    model_file_.fileName().clear();
    ui->lineEdit_modelfile->clear();
}

void GlobalView::on_radioButton_aci_clicked(){
    model_name_ = "aci";
    model_file_.fileName().clear();
    ui->lineEdit_modelfile->clear();
}

void GlobalView::on_radioButton_fct_clicked(){
    model_name_ = "fct";
    model_file_.fileName().clear();
    ui->lineEdit_modelfile->clear();
}

void GlobalView::on_radioButton_importfile_clicked(){
    if(ui->radioButton_importfile->isChecked()){
        importModel();
    }
    model_name_ = "";
}

/* Import an OpenQASM format file, representing the quantum circuit for a model. */
void GlobalView::importModel()
{
   QString fileName = QFileDialog::getOpenFileName(this, "Open file", globalDir+"/qasm_models");
   QFile file(fileName);
   model_file_ = QFileInfo(fileName);

   if (!file.open(QIODevice::ReadOnly)) {
       QMessageBox::warning(this, "Warning", "Unable to open the file: " + file.errorString());
       return;
   }else if(!fileName.endsWith(".qasm")){
       QMessageBox::warning(this, "Warning", "VeriQR only supports .qasm model files.");
       return;
   }
   ui->lineEdit_modelfile->setText(model_file_.filePath());

   file.close();
}

void GlobalView::on_radioButton_phaseflip_clicked()
{
    noise_type_ = noise_types[0];
    qDebug() << noise_type_;
}

void GlobalView::on_radioButton_bitflip_clicked()
{
    noise_type_ = noise_types[1];
    qDebug() << noise_type_;
}

void GlobalView::on_radioButton_depolarizing_clicked()
{
    noise_type_ = noise_types[2];
    qDebug() << noise_type_;
}

void GlobalView::on_radioButton_mixednoise_clicked()
{
    noise_type_ = noise_types[3];
    qDebug() << noise_type_;
}

void GlobalView::on_radioButton_importkraus_clicked()
{
    noise_type_ = "custom";
    QString fileName = QFileDialog::getOpenFileName(this, "Open file", globalDir+"/kraus");
    QFile file(fileName);
    kraus_file_ = QFileInfo(fileName);

    if (!file.open(QIODevice::ReadOnly)) {
        QMessageBox::warning(this, "Warning", "Unable to open the file: " + file.errorString());
        return;
    }else if(kraus_file_.suffix() != "npz"){
        QMessageBox::warning(this, "Warning", "VeriQR only supports .npz kraus files.");
        return;
    }

    ui->lineEdit_custom_noise->setText(kraus_file_.filePath());

    file.close();

    ui->radioButton_custom_noise->setChecked(true);
}

void GlobalView::run_calculate_k()
{
    clear_output();

    noise_prob_ = ui->doubleSpinBox_prob->value();

    QString cmd = "python";
    QStringList args;
    QString model_;
    if (model_file_.fileName().isEmpty())  // has not selected a .qasm file
    {
        model_ = model_name_;
    }
    else  // has selected a .qasm file
    {
        model_name_ = model_file_.fileName().chop(5);   // Remove '.qasm' in 'ehc_6.qasm'
        model_ = model_file_.filePath();
    }
    qDebug() << "model_name: " << model_name_;

    // python global_verif.py ehc_6.qasm phase_flip 0.0001
    args << cmd << pyfile_ << model_ << noise_type_ << QString::number(noise_prob_);
    QString paramsList = args.join(" ");
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

    csvfile_ = globalDir + "/results/result_tables/" + model_name_ + ".csv";
    QFile file(csvfile_);
    if(!file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        QMessageBox::warning(this, "Warning",
                             QString("Unable to open the file: %1 \n").arg(csvfile_)
                                 + file.errorString());
        return;
    }
    bool find_K = false;
    QTextStream in(&file);
    int row_index = 0;
    while (!in.atEnd()) {
        QString line = in.readLine();
        if(row_index > 0){
            if(line.contains(noise_type_.replace("_", "-") + "_" + QString::number(noise_prob_)))
            {
                find_K = true;
                lipschitz_ = line.split(',')[3].toDouble();
                verif_time_ = line.split(',')[4].toDouble();
                break;
            }
        }
        row_index++;
    }
    file.close();

    if(find_K)  // The current model has been verified before.
    {
        msgBox.exec();
        if(msgBox.clickedButton() == showButton){
            // show_saved_output(file);
            ui->lineEdit_k->setText(QString::number(lipschitz_));
            ui->lineEdit_time->setText(QString::number(verif_time_)+"s");
        }
        else{
            exec_calculation(cmd, args);
        }
    }
    else{
        exec_calculation(cmd, args);
    }

    // QString file = globalDir + "/results/runtime_output/" + file_name_ + ".txt";
    // if(findFile(file))  // The current model has been verified before.
    // {
    //     msgBox.exec();
    //     if(msgBox.clickedButton() == showButton){
    //         show_saved_output(file);
    //     }
    //     else{
    //         exec_calculation(cmd, args);
    //     }
    // }
    // else{
    //     exec_calculation(cmd, args);
    // }
}

void GlobalView::exec_calculation(QString cmd, QStringList args)
{
    process_cal = new QProcess(this);
    process_cal->setReadChannel(QProcess::StandardOutput);
    connect(process_cal, SIGNAL(stateChanged(QProcess::ProcessState)), SLOT(stateChanged(QProcess::ProcessState)));
    connect(process_cal, SIGNAL(readyReadStandardOutput()), this, SLOT(on_read_from_terminal_calc()));

    process_cal->setWorkingDirectory(globalDir);
    process_cal->start(cmd, args);
    if(!process_cal->waitForStarted()){
        qDebug() << "Process failure! Error: " << process_cal->errorString();
    }
    else{
        qDebug() << "Process succeed! ";
    }

    if (!process_cal->waitForFinished()) {
        qDebug() << "wait";
        QCoreApplication::processEvents(QEventLoop::AllEvents, 2000);
    }

    QString error = process_cal->readAllStandardError();  // Command line error message
    if(!error.isEmpty()){
        qDebug()<< "Error executing script： " << error;  // Printing error message
    }
}

void GlobalView::stateChanged(QProcess::ProcessState state)
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

void GlobalView::on_read_from_terminal_calc()
{
    while (process_cal->bytesAvailable() > 0){
        output_line_ = process_cal->readLine();
        output_.append(output_line_);
        ui->textBrowser_output->append(output_line_.simplified());
        //        qDebug() << output_line_;

        if(output_line_.contains(".svg was saved"))
        {
            show_circuit_diagram(globalDir + "/figures/" +
                                 output_line_.mid(0, output_line_.indexOf(".svg was saved")+4));
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
            verif_time_ = output_line_.mid(a, output_line_.size()-a-2).toDouble();
            ui->lineEdit_time->setText(QString::number(verif_time_)+"s");
            qDebug() << output_line_;
            qDebug() << verif_time_;
        }
        else if(output_line_.contains("The Global Verification End")){
            break;
        }
    }
}

// void GlobalView::show_loss_and_acc_plot()
// {
//     QString img_file = globalDir + "/loss_figures/" + file_name_ + ".png";
//     qDebug() << img_file;

//     QImage image(img_file);
//     QPixmap pixmap = QPixmap::fromImage(image);
//     // ui->imageLabel_plot->setPixmap(pixmap.scaled(ui->tabWidget->width()*0.8, ui->tabWidget->height()*0.8,
//     //                                              Qt::KeepAspectRatio, Qt::SmoothTransformation));
//     showed_loss = true;
// }

void GlobalView::show_circuit_diagram(QString img_file)
{
    QFile file(img_file);
    qDebug() << "show " << img_file;
    if(!file.open(QIODevice::ReadOnly))
    {
        QMessageBox::warning(this, "Warning", "Failed to open the file: " + img_file);
        return;
    }

    // QSpacerItem *verticalSpacer1 = new QSpacerItem(20, 40, QSizePolicy::Expanding, QSizePolicy::Expanding);
    // ui->verticalLayout_circ->addItem(verticalSpacer1);

    QFont font;
    font.setPointSize(12);
    font.setBold(true);
    if(img_file.contains("origin"))
    {
        QGroupBox *groupBox_origin_circ = new QGroupBox(ui->scrollArea_circ);
        groupBox_origin_circ->setObjectName("groupBox_origin_circ");
        groupBox_origin_circ->setTitle("Noiseless circuit");
        groupBox_origin_circ->setFont(font);
        ui->verticalLayout_circ->addWidget(groupBox_origin_circ, 1);
        QVBoxLayout *verticalLayout_origin = new QVBoxLayout(groupBox_origin_circ);
        verticalLayout_origin->setObjectName("verticalLayout_origin");

        SvgWidget *svgWidget = new SvgWidget(groupBox_origin_circ);
        svgWidget->load(img_file);
        svgWidget->setObjectName("svgWidget_origin");
        verticalLayout_origin->addWidget(svgWidget);
    }
    else if(img_file.contains("random"))
    {
        QGroupBox *groupBox_random_circ = new QGroupBox(ui->scrollArea_circ);
        groupBox_random_circ->setObjectName("groupBox_random_circ");
        groupBox_random_circ->setTitle("Circuit with random noise");
        groupBox_random_circ->setFont(font);
        ui->verticalLayout_circ->addWidget(groupBox_random_circ, 1);
        QVBoxLayout *verticalLayout_random = new QVBoxLayout(groupBox_random_circ);
        verticalLayout_random->setObjectName("verticalLayout_random");

        SvgWidget *svgWidget = new SvgWidget(groupBox_random_circ);
        svgWidget->load(img_file);
        svgWidget->setObjectName("svgWidget_random");
        verticalLayout_random->addWidget(svgWidget);
    }
    else
    {
        QGroupBox *groupBox_final_circ = new QGroupBox(ui->scrollArea_circ);
        groupBox_final_circ->setObjectName("groupBox_specified_circ");
        groupBox_final_circ->setTitle("Circuit with specified noise");
        groupBox_final_circ->setFont(font);
        ui->verticalLayout_circ->addWidget(groupBox_final_circ, 1);
        QVBoxLayout *verticalLayout_final = new QVBoxLayout(groupBox_final_circ);
        verticalLayout_final->setObjectName("verticalLayout_final");

        SvgWidget *svgWidget = new SvgWidget(groupBox_final_circ);
        svgWidget->load(img_file);
        svgWidget->setObjectName("svgWidget_final");
        verticalLayout_final->addWidget(svgWidget);
    }
    showed_svg = true;

    // qDebug() << "img_file: " << img_file;

    // SvgWidget *svgWidget = new SvgWidget(ui->scrollAreaWidgetContents_circ);
    // svgWidget->load(img_file);
    // svgWidget->setObjectName("svgWidget_circ");

    // double container_w = double(ui->scrollAreaWidgetContents_circ->width());
    // double svg_w = double(svgWidget->renderer()->defaultSize().width());
    // double svg_h = double(svgWidget->renderer()->defaultSize().height());
    // double iris_w = 977.0;
    // double iris_h = 260.0;
    // svg_h = container_w / svg_w * svg_h;
    // iris_h = container_w / iris_w * iris_h;
    // // qDebug() << svg_h;
    // // iris.svg: (width, height) = (977, 260)

    // QSpacerItem *verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Expanding, QSizePolicy::Expanding);
    // ui->verticalLayout_circ->addItem(verticalSpacer);
    // int diff = double(svg_h*2)/double(iris_h) * 1000;
    // // qDebug() << diff;
    // ui->verticalLayout_circ->insertWidget(0, svgWidget, diff);
    // ui->verticalLayout_circ->setStretch(1, 3*1000);

    // showed_svg = true;
}

void GlobalView::close_circuit_diagram()
{
    if(!showed_svg){
        return;
    }

    QWidget *svgwidget = ui->verticalLayout_circ->itemAt(0)->widget();
    svgwidget->setParent (NULL);
    qDebug() << "delete " << svgwidget->objectName() << "!";

    this->ui->verticalLayout_circ->removeWidget(svgwidget);
    delete svgwidget;

    showed_svg = false;
}

void GlobalView::stopProcess()
{
    this->process_cal->terminate();
    this->process_cal->waitForFinished();

    QMessageBox::information(this, "Notice", "The program was terminated.");
    qDebug() << "Process terminate!";
}

void GlobalView::run_globalVeri()
{
    if(epsilon_==0 || delta_==0 || lipschitz_==0){
        return;
    }

    // python global_verif.py verify k epsilon delta
    QString cmd = "python";
    QStringList args;
    args << cmd << pyfile_ << "verify" << QString::number(lipschitz_)
         << QString::number(epsilon_) << QString::number(delta_);
    QString paramsList = args.join(" ");
    qDebug() << paramsList;

    show_result_table();

    process_veri = new QProcess(this);
    process_veri->setReadChannel(QProcess::StandardOutput);
    connect(process_veri, SIGNAL(stateChanged(QProcess::ProcessState)), SLOT(stateChanged(QProcess::ProcessState)));
    connect(process_veri, SIGNAL(readyReadStandardOutput()), this, SLOT(on_read_from_terminal_verif()));

    process_veri->setWorkingDirectory(globalDir);
    process_veri->start(cmd, args);
    if(!process_veri->waitForStarted()){
        qDebug() << "Process failure! Error: " << process_veri->errorString();
    }
    else{
        qDebug() << "Process succeed! ";
    }

    if (!process_veri->waitForFinished()) {
        qDebug() << "wait";
        QCoreApplication::processEvents(QEventLoop::AllEvents, 2000);
    }

    QString error = process_veri->readAllStandardError();  // Command line error message
    if(!error.isEmpty()){
        qDebug()<< "Error executing script： " << error;  // Printing error message
    }
}

void GlobalView::on_read_from_terminal_verif()
{
    while (process_veri->bytesAvailable() > 0){
        output_line_ = process_veri->readLine();
        output_.append(output_line_);
        ui->textBrowser_output->append(output_line_.simplified());
        // qDebug() << output_line_;

        if(output_line_.contains("The Global Verification End")){
            // hf6_0.001_0.003_mixed_bitflip_0.13462
            QStringList noise_;
            if(noise_type_ == "mixed")
            {
                noise_ << noise_type_;
                mixed_noises_ = comboBox_mixednoise->current_select_items();
                for(int i = 0; i < mixed_noises_.count(); i++)
                {
                    // mixed_noises_[i] = mixed_noises_[i].replace(" ", "_");
                    noise_ << mixed_noises_[i].replace(" ", "");
                }
            }
            else if(noise_type_ == "custom")
            {
                QString krausfile = kraus_file_.filePath();
                noise_ << noise_type_ << krausfile;
            }
            else if(!noise_type_.isEmpty())  // bit flip or depolarizing or phase flip
            {
                noise_ << noise_type_.replace(" ", "");
            }

            QStringList list;
            list << model_name_.remove("_") << QString::number(epsilon_) << QString::number(delta_)
                 << noise_ << QString::number(noise_prob_);
            file_name_ = list.join("_");
            qDebug() << "file_name_: " << file_name_;

            // insert_data_to_table(res_count, );
            break;
        }
    }
}

GlobalView::~GlobalView()
{
    delete ui;
}

void GlobalView::on_slider_prob_sliderMoved(int pos)
{
    ui->doubleSpinBox_prob->setValue(ui->slider_prob->value()* 0.001);
    noise_prob_ = ui->doubleSpinBox_prob->value();
    qDebug() << "noise_prob: " << noise_prob_;
}

void GlobalView::on_doubleSpinBox_prob_valueChanged(double pos)
{
    ui->slider_prob->setValue(ui->doubleSpinBox_prob->value()/ 0.001);
    noise_prob_ = ui->doubleSpinBox_prob->value();
}

void GlobalView::on_slider_epsilon_sliderMoved(int pos)
{
    ui->doubleSpinBox_threshold_1->setValue(ui->slider_threshold_1->value()* 0.001);
    epsilon_ = ui->doubleSpinBox_threshold_1->value();
    qDebug() << "epsilon: " << epsilon_;
}

void GlobalView::on_doubleSpinBox_epsilon_valueChanged(double pos)
{
    ui->slider_threshold_1->setValue(ui->doubleSpinBox_threshold_1->value()/ 0.001);
    epsilon_ = ui->doubleSpinBox_threshold_1->value();
}

void GlobalView::on_slider_delta_sliderMoved(int pos)
{
    ui->doubleSpinBox_threshold_2->setValue(ui->slider_threshold_2->value()* 0.001);
    delta_ = ui->doubleSpinBox_threshold_2->value();
    qDebug() << "delta: " << delta_;
}

void GlobalView::on_doubleSpinBox_delta_valueChanged(double pos)
{
    ui->slider_threshold_2->setValue(ui->doubleSpinBox_threshold_2->value()/ 0.001);
    delta_ = ui->doubleSpinBox_threshold_2->value();
}

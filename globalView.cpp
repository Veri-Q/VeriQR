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
    connect(ui->pushButton_stop, SIGNAL(pressed()), this, SLOT(stop_process()));
    connect(ui->slider_epsilon, SIGNAL(valueChanged(int)), this, SLOT(on_slider_epsilon_sliderMoved(int)));
    connect(ui->doubleSpinBox_epsilon, SIGNAL(valueChanged(double)), this, SLOT(on_doubleSpinBox_epsilon_valueChanged(double)));
    connect(ui->slider_delta, SIGNAL(valueChanged(int)), this, SLOT(on_slider_delta_sliderMoved(int)));
    connect(ui->doubleSpinBox_delta, SIGNAL(valueChanged(double)), this, SLOT(on_doubleSpinBox_delta_valueChanged(double)));
    connect(ui->pushButton_veri, SIGNAL(pressed()), this, SLOT(run_globalVerify()));

    this->init();
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

    comboBox_mixednoise_ = new MultiSelectComboBox(ui->groupBox_noisetype);
    QSizePolicy sizePolicy1(QSizePolicy::Expanding, QSizePolicy::Fixed);
    sizePolicy1.setHorizontalStretch(0);
    sizePolicy1.setVerticalStretch(0);
    sizePolicy1.setHeightForWidth(comboBox_mixednoise_->sizePolicy().hasHeightForWidth());
    comboBox_mixednoise_->setSizePolicy(sizePolicy1);
    ui->gridLayout->addWidget(comboBox_mixednoise_, 0, 5, 1, 1);
    QStringList noiseList;
    noiseList << "bit flip" << "depolarizing" << "phase flip";
    comboBox_mixednoise_->setMaxSelectNum(3);
    comboBox_mixednoise_->addItems_for_noise(noiseList);

    // show_result_table();
}

void GlobalView::resizeEvent(QResizeEvent *)
{
    // if(showed_loss)
    // {
    //     show_loss_and_acc_plot();
    // }
}

bool GlobalView::fileExists(QString filename)
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
        qDebug() << "Found the file: " << filename;
        return true;
    }

    qDebug() << "The file don't exist! ";
    return false;
}

void GlobalView::clearOutput()
{
    output_.clear();
    ui->textBrowser_output->clear();

    res_model_ = new QStandardItemModel();
    got_result_ = false;
    calc_count_ = 0;
    verif_count_ = 0;

    origin_lipschitz_ = 0.0;
    random_lipschitz_ = 0.0;
    specified_lipschitz_ = 0.0;
    origin_VT_ = 0.0;
    random_VT_ = 0.0;
    specified_VT_ = 0.0;
    origin_robust_.clear();
    random_robust_.clear();
    specified_robust_.clear();

    closeCircuitDiagram();
}

void GlobalView::resetAll()
{
    clearOutput();

    // Reset the options in the interface and related variables.
    // Basic settings:
    model_file_ = QFileInfo();
    model_name_.clear();
    file_name_.clear();
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
    comboBox_mixednoise_->line_edit_->clear();
    for(int i = 0; i < comboBox_mixednoise_->list_widget_->count(); i++)
    {
        QCheckBox *noise_check_box = static_cast<QCheckBox*>(
            comboBox_mixednoise_->list_widget_->itemWidget(
                comboBox_mixednoise_->list_widget_->item(i)
                )
            );
        noise_check_box->setChecked(0);
    }
    kraus_file_ = QFileInfo();
    ui->lineEdit_custom_noise->clear();
    ui->radioButton_custom_noise->setChecked(0);

    noise_prob_ = 0.0;
    ui->doubleSpinBox_prob->setValue(0.0);
    ui->slider_prob->setValue(0);

    epsilon_ = 0.0;
    ui->doubleSpinBox_epsilon->setValue(0.0);
    ui->slider_epsilon->setValue(0);

    delta_ = 0.0;
    ui->doubleSpinBox_delta->setValue(0.0);
    ui->slider_delta->setValue(0);

    result_dir_.clear();
    csvfile_.clear();
    txtfile_.clear();

    update();
    qDebug() << "Reset all settings! ";
}

/* Open a csv file that saves verification results. */
void GlobalView::openCsvfile(){
    QString fileName = QFileDialog::getOpenFileName(this, "Open file", globalDir+"/results");

    showResultFromCsvfile(fileName);
}

void GlobalView::showResultFromCsvfile(QString fileName)
{
    clearOutput();
    readLipschitzFromCsvfile(fileName);

    csvfile_ = fileName;
    // e.g. cr_mixed_BitFlip_Depolarizing_PhaseFlip_0.005.csv
    file_name_ = fileName.mid(fileName.lastIndexOf("/")+1, fileName.indexOf(".csv"));
    file_name_.chop(4);
    result_dir_ = fileName.mid(0, fileName.lastIndexOf("/"));
    qDebug() << "result_dir_: " << result_dir_;
    qDebug() << "file_name_: " << file_name_;
    qDebug() << "csvfile_: " << csvfile_;

    // Obtain parameters from the filename
    QStringList args = file_name_.split("_");

    // Selected model change
    int arg_start_index = 0;
    if(file_name_.startsWith("cr")){
        model_name_ = args[0];
        ui->radioButton_cr->setChecked(1);
    }
    else if(file_name_.startsWith("aci")){
        model_name_ = args[0];
        ui->radioButton_aci->setChecked(1);
    }
    else if(file_name_.startsWith("fct")){
        model_name_ = args[0];
        ui->radioButton_fct->setChecked(1);
    }
    else{
        model_name_ = args[0] + "_" + args[1];  // args[0] + "_" + args[1]
        arg_start_index = 1;
        ui->radioButton_importfile->setChecked(1);
        ui->lineEdit_modelfile->setText(globalDir+"/qasm_models/"+model_name_+".qasm");
    }

    // epsilon_ = args[arg_start_index + 1].toDouble();
    // delta_ = args[arg_start_index + 2].toDouble();
    // Noise settings change
    noise_type_ = args[arg_start_index + 1];
    qDebug() << noise_type_;
    if(noise_type_ == "PhaseFlip"){
        ui->radioButton_phaseflip->setChecked(1);
    }
    else if(noise_type_ == "BitFlip"){
        ui->radioButton_bitflip->setChecked(1);
    }
    else if(noise_type_ == "Depolarizing"){
        ui->radioButton_depolarizing->setChecked(1);
    }
    else if(noise_type_ == "mixed"){
        ui->radioButton_mixednoise->setChecked(1);
        for(int i = arg_start_index + 2; i < args.size()-1; i++)
        {
            QString noise = noise_name_map_1[args[i]];
            mixed_noises_.append(noise);
        }
        comboBox_mixednoise_->line_edit_->setText(mixed_noises_.join(";"));
    }
    else if(noise_type_ == "custom")  // hf_6_custom_kraus1qubit_0.001
    {
        kraus_file_ = QFileInfo(globalDir + "/kraus/" + args[arg_start_index + 2] + ".npz");
        ui->lineEdit_custom_noise->setText(kraus_file_.filePath());
    }

    noise_prob_ = args[args.size()-1].toDouble();
    ui->doubleSpinBox_prob->setValue(noise_prob_);
    qDebug() << noise_prob_;

    showCircuitDiagram(result_dir_ + "/" + model_name_ + "_origin.svg");
    showCircuitDiagram(result_dir_ + "/" + model_name_ + "_random.svg");
    showCircuitDiagram(result_dir_ + "/" + file_name_ + ".svg");
}

void GlobalView::showOutputFromTxtfile(QString fileName)
{
    QFile txtfile(fileName);
    if(!txtfile.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QMessageBox::warning(this, "Warning", "Unable to open the file " + fileName + ": "
                                                  + txtfile.errorString());
        return;
    }

    clearOutput();

    // e.g. hf_6_0.001_0.003_mixed_bitflip_0.13462.txt
    // e.g. cr_0.001_0.003_mixed_bitflip_0.13462.txt
    file_name_ = fileName.mid(fileName.lastIndexOf("/")+1, fileName.lastIndexOf("."));
    result_dir_ = globalDir + "/results/" + model_name_ + "/" + file_name_;
    txtfile_ = fileName;
    csvfile_ = txtfile_.replace("txt", "csv");
    qDebug() << "result_dir_: " << result_dir_;
    qDebug() << "file_name_: " << file_name_;
    qDebug() << "txtfile_: " << txtfile_;
    qDebug() << "csvfile_: " << csvfile_;

    // Obtain parameters from the filename
    QStringList args = file_name_.split("_");

    // Selected model change
    int arg_start_index = 0;
    if(file_name_.startsWith("cr")){
        model_name_ = args[0];
        ui->radioButton_cr->setChecked(1);
    }
    else if(file_name_.startsWith("aci")){
        model_name_ = args[0];
        ui->radioButton_aci->setChecked(1);
    }
    else if(file_name_.startsWith("fct")){
        model_name_ = args[0];
        ui->radioButton_fct->setChecked(1);
    }
    else{
        model_name_ = args[0] + "_" + args[1];  // args[0] + "_" + args[1]
        arg_start_index = 1;
        ui->radioButton_importfile->setChecked(1);
        ui->lineEdit_modelfile->setText(globalDir+"/qasm_models/"+model_name_+".qasm");
    }

    epsilon_ = args[arg_start_index + 1].toDouble();
    delta_ = args[arg_start_index + 2].toDouble();
    // Noise settings change
    noise_prob_ = args[args.size()-1].toDouble();
    ui->doubleSpinBox_prob->setValue(noise_prob_);
    noise_type_ = args[arg_start_index + 3];
    if(noise_type_ == "PhaseFlip"){
        ui->radioButton_phaseflip->setChecked(1);
    }
    else if(noise_type_ == "BitFlip"){
        ui->radioButton_bitflip->setChecked(1);
    }
    else if(noise_type_ == "Depolarizing"){
        ui->radioButton_depolarizing->setChecked(1);
    }
    else if(noise_type_ == "mixed"){
        ui->radioButton_mixednoise->setChecked(1);
        for(int i = arg_start_index + 4; i < args.size()-1; i++)
        {
            QString noise = noise_name_map_1[args[i]];
            mixed_noises_.append(noise);
        }
        comboBox_mixednoise_->line_edit_->setText(mixed_noises_.join(";"));
    }
    else if(noise_type_ == "custom")  // hf6_0.001_0.003_custom_kraus1qubit_0.001
    {
        kraus_file_ = QFileInfo(globalDir + "/kraus/" + args[arg_start_index + 4] + ".npz");
        ui->lineEdit_custom_noise->setText(kraus_file_.filePath());
    }
    qDebug() << noise_prob_;
    qDebug() << noise_type_;

    output_ = QString::fromLocal8Bit(txtfile.readAll());
    ui->textBrowser_output->setText(output_);
    txtfile.close();

    readLipschitzFromCsvfile(csvfile_);

    showCircuitDiagram(result_dir_ + "/" + model_name_ + "_origin.svg");
    showCircuitDiagram(result_dir_ + "/" + model_name_ + "_random.svg");
    showCircuitDiagram(result_dir_ + "/" + file_name_ + ".svg");
}

void GlobalView::readLipschitzFromCsvfile(QString fileName)
{
    QFile csvfile(fileName);
    if(!csvfile.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        QMessageBox::warning(this, "Warning", "Unable to open the file " + fileName + ": "
                                                  + csvfile.errorString());
        return;
    }
    else if(QFileInfo(fileName).suffix() != "csv")
    {
        QMessageBox::warning(this, "Warning", "The suffix of the file is '"
                                                  + QFileInfo(fileName).suffix() + "', not 'csv'");
        return;
    }

    // Initial table data.
    showResultTable();

    QTextStream in(&csvfile);
    int row_index = -1;
    while (!in.atEnd()) {
        QString line = in.readLine();
        if(row_index >= 0){
            QStringList res_fields = line.split(",");
            QString perturbations = QString("%1, %2").arg(QString(res_fields[1]),
                                                          QString(res_fields[2]));
            insertLineDataToTable(row_index,
                                  QString(res_fields[0]),
                                  perturbations.remove('"'),
                                  QString(res_fields[3]),
                                  QString(res_fields[4]),
                                  QString(res_fields[5]));
            if(row_index % 3 == 2)
            {
                ui->table_res->setSpan(row_index-2, 1, 3, 1);
            }
        }
        row_index++;
    }
    csvfile.close();
}

void GlobalView::readLipschitzFromTable()
{
    origin_lipschitz_ = res_model_->index(0, 2).data().toString().toDouble();
    random_lipschitz_ = res_model_->index(1, 2).data().toString().toDouble();
    specified_lipschitz_ = res_model_->index(2, 2).data().toString().toDouble();
    origin_VT_ = res_model_->index(0, 3).data().toString().toDouble();
    random_VT_ = res_model_->index(1, 3).data().toString().toDouble();
    specified_VT_ = res_model_->index(2, 3).data().toString().toDouble();
    got_result_ = true;
}

bool GlobalView::getVerifResultFromCsvfile(QString fileName, double epsilon, double delta)
{
    QFile csvfile(fileName);
    if(!csvfile.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        // QMessageBox::information(this, "Notice", "Unable to open the file " + fileName + ": "
        //                                           + csvfile.errorString());
        qDebug() << "Unable to open the file " + fileName + ": " + csvfile.errorString();
        return false;
    }

    QString perturbations = QString("%1, %2").arg(QString::number(epsilon), QString::number(delta));
    QTextStream in(&csvfile);
    int row_index = -1;
    while (!in.atEnd()) {
        QString line = in.readLine();
        if(row_index >= 0){
            QStringList res_fields = line.split(",");
            QString perturbations_ = QString("%1, %2").arg(QString(res_fields[1]),
                                                          QString(res_fields[2]));
            if(perturbations_ != perturbations){
                return false;
            }
            else{
                if(row_index == 0){
                    origin_robust_ = QString(res_fields[5]);
                }
                else if(row_index == 1){
                    random_robust_ = QString(res_fields[5]);
                }
                else if(row_index == 2){
                    specified_robust_ = QString(res_fields[5]);
                    break;
                }
            }
        }
        row_index++;
    }
    csvfile.close();
    return true;
}

void GlobalView::showResultTable(){
    res_model_ = new QStandardItemModel();
    // Set the column header.
    res_model_->setHorizontalHeaderLabels(QStringList() << "Circuit (noise_p)" << "Perturbations (ε, δ)"
                                                       << "K*" << "VT(s)" << "If Robust");

    // Add QStandardItemModel to QTableView.
    ui->table_res->setModel(res_model_);

    QString header_style = "QHeaderView::section{"
                           "background:rgb(120,120,120);"
                           "color:rgb(255,255,255);"
                           "padding: 1px;}";
    ui->table_res->setShowGrid(true);
    ui->table_res->setGridStyle(Qt::DotLine);
    ui->table_res->verticalHeader()->setHidden(true);  // Remove the automatic serial number column.
    ui->table_res->verticalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    ui->table_res->verticalHeader()->setStyleSheet(header_style);
    ui->table_res->horizontalHeader()->setSectionResizeMode(QHeaderView::Interactive);
    ui->table_res->horizontalHeader()->setStyleSheet(header_style);
    ui->table_res->horizontalHeader()->setMinimumHeight(50);
    ui->table_res->setColumnWidth(0, ui->table_res->width()/8*2.8);
    ui->table_res->setColumnWidth(1, ui->table_res->width()/8*2);
    ui->table_res->setColumnWidth(2, ui->table_res->width()/8*1.2);
    ui->table_res->setColumnWidth(3, ui->table_res->width()/8*1);
    ui->table_res->setColumnWidth(4, ui->table_res->width()/8*1);

    insertLineDataToTable(calc_count_*3, QString("noiseless"),
                          QString("-"), QString("-"), QString("-"), QString("-"));
    insertLineDataToTable(calc_count_*3+1, QString("random noise"),
                          QString("-"), QString("-"), QString("-"), QString("-"));
    insertLineDataToTable(calc_count_*3+2,
                          QString("random & %1_%2").arg(noise_type_.replace("_", "-"),
                                                        QString::number(noise_prob_)),
                          QString("-"), QString("-"), QString("-"), QString("-"));
}

void GlobalView::insertDataToTable(int row_index, int col_index, QString data)
{
    QStandardItem *item = new QStandardItem(data);
    item->setTextAlignment(Qt::AlignCenter);
    res_model_->setItem(row_index, col_index, item);
}

void GlobalView::insertLineDataToTable(int row_index, QString circuit, QString perturbations,
                                      QString K, QString VT, QString if_robust)
{
    insertDataToTable(row_index, 0, circuit);
    insertDataToTable(row_index, 1, perturbations);
    insertDataToTable(row_index, 2, K);
    insertDataToTable(row_index, 3, VT);
    insertDataToTable(row_index, 4, if_robust);
}

// /* Open a txt file that records the runtime output. (deprecated)*/
// void GlobalView::show_saved_output_1(QString fileName)
// {
//     // QString fileName = QFileDialog::getOpenFileName(this, "Open file", globalDir+"/results/runtime_output");
//     QFile file(fileName);

//     if (!file.open(QIODevice::ReadOnly |QIODevice::Text)) {
//         QMessageBox::warning(this, "Warning", "Unable to open the file: " + file.errorString());
//         return;
//     }
//     else if(QFileInfo(fileName).suffix() != "txt") {
//         QMessageBox::warning(this, "Warning", "VeriQR only supports .txt output information files.");
//         return;
//     }

//     reset_all();

//     output_ = QString::fromLocal8Bit(file.readAll());
//     ui->textBrowser_output->setText(output_);
//     file.close();

//     file_name_ = QFileInfo(file).fileName();  // e.g. hf6_0.001_0.003_mixed_bitflip_0.13462.txt
//     file_name_.chop(4);
//     qDebug() << "file_name_: " << file_name_;

//     // Obtain parameters from the filename
//     QStringList args = file_name_.split("_");
//     model_name_ = args[0];
//     epsilon_ = args[1].toDouble();
//     delta_ = args[2].toDouble();
//     // model_name_ = file_name_.mid(0, file_name_.indexOf(args[1])-1);
//     int noise_type_index = 3;
//     noise_type_ = args[noise_type_index];
//     noise_prob_ = args[args.size()-1].toDouble();
//     ui->doubleSpinBox_prob->setValue(noise_prob_);

//     // Noise settings change
//     if(noise_type_ == "phaseflip"){
//         ui->radioButton_phaseflip->setChecked(1);
//         noise_type_ = "phase_flip";
//     }
//     else if(noise_type_ == "bitflip"){
//         ui->radioButton_bitflip->setChecked(1);
//         noise_type_ = "bit_flip";
//     }
//     else if(noise_type_ == "depolarizing"){
//         ui->radioButton_depolarizing->setChecked(1);
//         noise_type_ = "depolarizing";
//         for(int i = noise_type_index + 1; i < args.size()-1; i++)
//         {
//             QString noise = noise_name_map_1[args[i]];
//             mixed_noises_.append(noise);
//         }
//         comboBox_mixednoise_->line_edit_->setText(mixed_noises_.join(";"));
//     }
//     else if(noise_type_ == "mixed"){
//         ui->radioButton_mixednoise->setChecked(1);
//     }
//     else if(noise_type_ == "custom")  // hf6_0.001_0.003_custom_kraus1qubit_0.001
//     {
//         kraus_file_ = QFileInfo(globalDir + "/kraus/" + args[4] + ".npz");
//         ui->lineEdit_custom_noise->setText(kraus_file_.filePath());
//     }
//     qDebug() << noise_prob_;
//     qDebug() << noise_type_;

//     // Selected model change
//     if(file_name_.startsWith("cr")){
//         ui->radioButton_cr->setChecked(1);
//     }
//     else if(file_name_.startsWith("aci")){
//         ui->radioButton_aci->setChecked(1);
//     }
//     else if(file_name_.startsWith("fct")){
//         ui->radioButton_fct->setChecked(1);
//     }
//     else{
//         ui->radioButton_importfile->setChecked(1);
//         // ui->lineEdit_modelfile->setText(globalDir+"/qasm_models/"+model_name_+".qasm");
//     }

//     while (!file.atEnd())
//     {
//         QString line = QString(file.readLine());
//         output_.append(line);
//         ui->textBrowser_output->append(line.simplified());
//         // qDebug() << line;

//         if(line.contains(".svg was saved"))
//         {
//             showCircuitDiagram(globalDir + "/figures/" +
//                                  line.mid(0, line.indexOf(".svg was saved")+4));
//         }
//         else if(line.startsWith("Lipschitz K"))
//         {
//             origin_lipschitz_ = line.mid(line.indexOf("= ")+2).toDouble();
//             // ui->lineEdit_k->setText(QString::number(lipschitz_));
//             qDebug() << line;
//             qDebug() << origin_lipschitz_;
//         }
//         else if(line.startsWith("Elapsed time"))
//         {
//             int a = line.indexOf("= ") + 2;
//             origin_VT_ = line.mid(a, line.size()-a-2).toDouble();
//             // ui->lineEdit_time->setText(QString::number(verif_time_)+"s");
//             qDebug() << line;
//             qDebug() << origin_VT_;
//         }
//         else if(line.contains("The Global Verification End")){
//             break;
//         }
//     }

//     file.close();
// }

/* Save the runtime output as a txt file to the specified location. */
void GlobalView::saveTableToCsvfile()
{
    if(!got_result_)
    {
        QMessageBox::warning(this, "Warning", "The computation of the Lipschitz constant "
                                              "for the three circuits is not finished; "
                                              "you have to wait for the computation or "
                                              "recalculate.");
        return;
    }

    QFile file(csvfile_);
    if(file.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Truncate)){
        QTextStream stream(&file);
        stream << "Circuit" << "(ε δ)" << "K*" << "VT (s)" << "robust" << "\n";
        for(int row_index = 0; row_index < res_model_->rowCount(); row_index++)
        {
            QString noise = res_model_->index(row_index, 0).data().toString() + ",";
            QString per = res_model_->index(row_index, 1).data().toString() + ",";
            QString k = res_model_->index(row_index, 2).data().toString() + ",";
            QString vt = res_model_->index(row_index, 3).data().toString() + ",";
            QString robust = res_model_->index(row_index, 4).data().toString();
            stream << noise << per << k << vt << robust << "\n";
        }
        file.close();
        QMessageBox::information(NULL, "",
                                 QString("The file %1 was saved successfully!").arg(csvfile_),
                                 QMessageBox::Yes);
    }
}

void GlobalView::saveTableAsFile()
{
    if(!got_result_)
    {
        QMessageBox::warning(this, "Warning", "The computation of the Lipschitz constant "
                                              "for the three circuits is not finished; "
                                              "you have to wait for the computation or "
                                              "recalculate.");
        return;
    }

    QString fileName = QFileDialog::getSaveFileName(this, "Save as", result_dir_);
    QFile file(fileName);
    if (!file.open(QFile::WriteOnly | QFile::Text)) {
        QMessageBox::warning(this, "Warning", "Unable to open the file " + fileName + ": "
                                                  + file.errorString());
        return;
    }

    QTextStream stream(&file);
    for(int row_index = 0; row_index < res_model_->rowCount(); row_index++)
    {
        QString noise = res_model_->index(row_index, 0).data().toString() + ",";
        QString per = res_model_->index(row_index, 1).data().toString() + ",";
        QString k = res_model_->index(row_index, 2).data().toString() + ",";
        QString vt = res_model_->index(row_index, 3).data().toString() + ",";
        QString robust = res_model_->index(row_index, 4).data().toString();
        stream << noise << per << k << vt << robust << "\n";
    }
    file.close();
    QMessageBox::information(NULL, "",
                             QString("The file %1 was saved successfully!").arg(csvfile_),
                             QMessageBox::Yes);
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

void GlobalView::on_radioButton_importfile_clicked()
{
    if(ui->radioButton_importfile->isChecked()){
        importModel();
    }
}

/* Import an OpenQASM format file, representing the quantum circuit for a model. */
void GlobalView::importModel()
{
   QString fileName = QFileDialog::getOpenFileName(this, "Open file", globalDir+"/qasm_models");
   QFile file(fileName);
   model_file_ = QFileInfo(fileName);

   if (!file.open(QIODevice::ReadOnly)) {
       QMessageBox::warning(this, "Warning", "Unable to open the file " + fileName + ": "
                                                 + file.errorString());
       return;
   }else if(!fileName.endsWith(".qasm")){
       QMessageBox::warning(this, "Warning", "VeriQR only supports .qasm model files.");
       return;
   }

   ui->lineEdit_modelfile->setText(model_file_.filePath());

   model_name_ = fileName.mid(fileName.lastIndexOf("/")+1);
   model_name_.chop(5);
   qDebug() << "model_name_: " << model_name_;

   file.close();
}

void GlobalView::on_radioButton_bitflip_clicked()
{
    noise_type_ = noise_types_[0];
    qDebug() << noise_type_;
}

void GlobalView::on_radioButton_depolarizing_clicked()
{
    noise_type_ = noise_types_[1];
    qDebug() << noise_type_;
}

void GlobalView::on_radioButton_phaseflip_clicked()
{
    noise_type_ = noise_types_[2];
    qDebug() << noise_type_;
}

void GlobalView::on_radioButton_mixednoise_clicked()
{
    noise_type_ = noise_types_[3];
    qDebug() << noise_type_;
}

void GlobalView::on_radioButton_importkraus_clicked()
{
    noise_type_ = "custom";
    QString fileName = QFileDialog::getOpenFileName(this, "Open file", globalDir+"/kraus");
    QFile file(fileName);
    kraus_file_ = QFileInfo(fileName);

    if (!file.open(QIODevice::ReadOnly)) {
        QMessageBox::warning(this, "Warning", "Unable to open the file " + fileName + ": "
                                                  + file.errorString());
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
    clearOutput();

    noise_prob_ = ui->doubleSpinBox_prob->value();

    QString cmd = "python";
    QString model_file_name = model_name_;  // ehc_6
    if(!model_file_.fileName().isEmpty())  // has selected a .qasm file
    {
        model_file_name = model_file_.filePath();
    }
    qDebug() << "model_name_: " << model_name_;
    qDebug() << "model_file_name: " << model_file_name;

    // python global_verif.py ehc_6.qasreadLipschitzFromCsvfilem phase_flip 0.0001
    QStringList args;
    args << model_file_name;
    if(noise_type_ == "mixed")
    {
        args << noise_type_;
        mixed_noises_ = comboBox_mixednoise_->current_select_items();
        for(int i = 0; i < mixed_noises_.count(); i++)
        {
            // mixed_noises_[i] = mixed_noises_[i].replace(" ", "_");
            args << mixed_noises_[i].replace(" ", "_");
        }
        args << QString::number(noise_prob_);
    }
    else if(noise_type_ == "custom")
    {
        QString krausfile = kraus_file_.filePath();
        args << noise_type_ << krausfile << QString::number(noise_prob_);
    }
    else if(!noise_type_.isEmpty())  // bit flip or depolarizing or phase flip
    {
        args << noise_type_ << QString::number(noise_prob_);
    }

    file_name_ = args.join("_");
    qDebug() << "file_name_: " << file_name_;
    for(auto it=noise_name_map_2.begin(); it!=noise_name_map_2.end(); it++)
    {
        file_name_.replace(it.key(), it.value());
    }
    qDebug() << "file_name_: " << file_name_.replace(model_file_name, model_name_);

    args.insert(0, pyfile_);
    qDebug() << args.join(" ");

    // QStandardItem *item_c0 = new QStandardItem(QString("noiseless"));
    // QStandardItem *item_c1 = new QStandardItem(QString("random noise"));
    // QStandardItem *item_c2 = new QStandardItem(QString("random & %1_%2").arg(noise_type_.replace("_", "-"),
    //                                                                          QString::number(noise_prob_)));
    // item_c0->setTextAlignment(Qt::AlignCenter);
    // item_c1->setTextAlignment(Qt::AlignCenter);
    // item_c2->setTextAlignment(Qt::AlignCenter);
    // res_model->setItem(res_count_*3+1, 0, item_c0);
    // res_model->setItem(res_count_*3+2, 0, item_c1);
    // res_model->setItem(res_count_*3+3, 0, item_c2);

    // for(int row_index = 0; row_index < 3; row_index++)
    // {
    //     for(int col_index = 1; col_index < 5; col_index++)
    //     {
    //         QStandardItem *item = new QStandardItem(QString(""));
    //         item->setTextAlignment(Qt::AlignCenter);
    //         res_model->setItem(row_index, col_index, item);
    //     }
    // }

    result_dir_ = globalDir + "/results/" + model_name_ + "/" + file_name_;
    csvfile_ = result_dir_ + "/" + file_name_ + ".csv";
    // txtfile_ = result_dir_ + "/" + file_name_ + ".txt";
    qDebug() << "result_dir_: " << result_dir_;
    qDebug() << "csvfile_: " << csvfile_;
    if(fileExists(result_dir_) && fileExists(csvfile_))  // The current model has been verified before.
    {
        QString strInfo = "The results of this model have been saved before. "
                          "Do you want to see the previous results or recalculate it?";
        QMessageBox msgBox;
        msgBox.setWindowTitle("Whether to Recalculate the Lipschitz");
        msgBox.setText(strInfo);
        QPushButton *showButton = msgBox.addButton(tr("Show the previous results"), QMessageBox::ActionRole);
        msgBox.addButton(tr("Recalculate"),QMessageBox::ActionRole);
        msgBox.addButton(QMessageBox::No);
        msgBox.button(QMessageBox::No)->setHidden(true);
        msgBox.setDefaultButton(QMessageBox::NoButton);
        msgBox.exec();
        if(msgBox.clickedButton() == showButton)
        {
            // QString fileName = QFileDialog::getOpenFileName(this, "Open file", result_dir_);
            readLipschitzFromCsvfile(csvfile_);
            readLipschitzFromTable();
            showCircuitDiagram(result_dir_ + "/" + model_name_ + "_origin.svg");
            showCircuitDiagram(result_dir_ + "/" + model_name_ + "_random.svg");
            showCircuitDiagram(result_dir_ + "/" + model_name_ + ".svg");
        }
        else{
            // Initial table data.
            showResultTable();
            execCalculation(cmd, args);
        }
    }
    else{
        QString model_dir = globalDir + "/results/" + model_name_;
        QDir dir(model_dir);
        if(!dir.exists())
        {
            // Create a folder named 'model_name' for the model if it doesn't exist
            dir.mkdir(model_dir);
            qDebug() << "The folder " << model_dir << " created successfully!";
            QDir dir(result_dir_);
            dir.mkdir(result_dir_);
        }
        else
        {
            QDir dir(result_dir_);
            if(!dir.exists())
            {
                dir.mkdir(result_dir_);
            }
        }
        // Initial table data.
        showResultTable();
        execCalculation(cmd, args);
    }
}

void GlobalView::execCalculation(QString cmd, QStringList args)
{
    process_cal_ = new QProcess(this);
    process_cal_->setReadChannel(QProcess::StandardOutput);
    connect(process_cal_, SIGNAL(computationStateChanged(QProcess::ProcessState)), SLOT(computationStateChanged(QProcess::ProcessState)));
    connect(process_cal_, SIGNAL(readyReadStandardOutput()), this, SLOT(on_read_from_terminal_calc()));

    process_cal_->setWorkingDirectory(globalDir);
    process_cal_->start(cmd, args);
    if(!process_cal_->waitForStarted()){
        qDebug() << "Process failure! Error: " << process_cal_->errorString();
    }
    else{
        qDebug() << "Process succeed! ";
    }

    if (!process_cal_->waitForFinished()) {
        qDebug() << "wait";
        QCoreApplication::processEvents(QEventLoop::AllEvents, 2000);
    }

    QString error = process_cal_->readAllStandardError();  // Command line error message
    if(!error.isEmpty()){
        qDebug()<< "Error executing script： " << error;  // Printing error message
    }
}

void GlobalView::on_read_from_terminal_calc()
{
    while(process_cal_->bytesAvailable() > 0){
        QString line = process_cal_->readLine();
        output_.append(line);
        ui->textBrowser_output->append(line.simplified());
        // qDebug() << line;

        line = line.simplified();
        if(line.contains(".svg was saved"))
        {
            showCircuitDiagram(result_dir_ + "/" + line.mid(0, line.indexOf(".svg")+4));
        }
        else if(line.startsWith("Lipschitz K"))
        {
            QString k = line.mid(line.indexOf("= ")+2);
            // origin_lipschitz_ = line.mid(line.indexOf("= ")+2).toDouble();
            // ui->lineEdit_k->setText(QString::number(lipschitz_));
            qDebug() << line;
            qDebug() << k;
            // QStandardItem *item = new QStandardItem(k);
            // item->setTextAlignment(Qt::AlignCenter);
            // res_model_->setItem(res_count_, 2, item);
            insertDataToTable(calc_count_, 2, k);
        }
        else if(line.startsWith("Elapsed time"))
        {
            QString vt = line.mid(line.indexOf("= ") + 2);
            // origin_VT_ = line.mid(a, line.size()-a-2).toDouble();
            qDebug() << line;
            qDebug() << vt;
            // QStandardItem *item = new QStandardItem(vt);
            // item->setTextAlignment(Qt::AlignCenter);
            // res_model->setItem(res_count_, 3, item);
            insertDataToTable(calc_count_, 3, vt);
        }
        else if(line.contains("The Lipschitz Constant Calculation End")){
            calc_count_++;
            if(calc_count_ % 3 == 0)
            {
                readLipschitzFromTable();
                qDebug() << "origin_lipschitz_： " << origin_lipschitz_;
                qDebug() << "random_lipschitz_ " << random_lipschitz_;
                qDebug() << "specified_lipschitz_ " << specified_lipschitz_;
                break;
            }
        }
    }
}

void GlobalView::computationStateChanged(QProcess::ProcessState state)
{
    qDebug() << "show state:";
    switch(state)
    {
    case QProcess::NotRunning:
        qDebug() << "Not Running";
        break;
    case QProcess::Starting:
        qDebug() << "Starting";
        break;
    case QProcess::Running:
        qDebug() << "Running";
        break;
    default:
        qDebug() << "otherState";
        break;
    }
}

void GlobalView::verificationStateChanged(QProcess::ProcessState state)
{
    qDebug() << "show state:";
    switch(state)
    {
    case QProcess::NotRunning:
        qDebug() << "Not Running";
        break;
    case QProcess::Starting:
        qDebug() << "Starting";
        break;
    case QProcess::Running:
        qDebug() << "Running";
        break;
    default:
        qDebug() << "otherState";
        break;
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

void GlobalView::showCircuitDiagram(QString img_file)
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
    showed_svg_ = true;
}

void GlobalView::closeCircuitDiagram()
{
    if(!showed_svg_){
        return;
    }

    QLayoutItem *child;
    int i = 0;
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
    showed_svg_ = false;
}

void GlobalView::stop_process()
{
    this->process_cal_->terminate();
    this->process_cal_->waitForFinished();

    QMessageBox::information(this, "Notice", "The program was terminated.");
    qDebug() << "Process terminate!";
}

void GlobalView::global_verify(double epsilon, double delta, QList<double> kList)
{
    if(epsilon==0.0 || delta==0.0){
        QMessageBox::warning(this, "Warning", "The perturbation parameter cannot be 0.");
        return;
    }
    // python global_verif.py verify k epsilon delta
    QString cmd = "python";
    QStringList args;
    args << pyfile_ << "verify";
    for(int i=0; i < kList.size(); i++)
    {
        if(kList.at(i) == 0.0)
        {
            QMessageBox::warning(this, "Warning", "The Lipschitz constant cannot be 0.");
            return;
        }
        else
        {
            args << QString::number(kList.at(i));
        }
    }
    args << QString::number(epsilon) << QString::number(delta);
    qDebug() << cmd + " " + args.join(" ");

    process_veri_ = new QProcess(this);
    process_veri_->setReadChannel(QProcess::StandardOutput);
    connect(process_veri_, SIGNAL(verificationStateChanged(QProcess::ProcessState)),
            SLOT(verificationStateChanged(QProcess::ProcessState)));
    connect(process_veri_, SIGNAL(readyReadStandardOutput()), this,
            SLOT(on_read_from_terminal_verif()));

    process_veri_->setWorkingDirectory(globalDir);
    process_veri_->start(cmd, args);
    if(!process_veri_->waitForStarted()){
        qDebug() << "Process failure! Error: " << process_veri_->errorString();
    }
    else{
        qDebug() << "Process succeed! ";
    }

    if (!process_veri_->waitForFinished()) {
        qDebug() << "wait";
        QCoreApplication::processEvents(QEventLoop::AllEvents, 2000);
    }

    QString error = process_veri_->readAllStandardError();  // Command line error message
    if(!error.isEmpty()){
        qDebug() << "Error executing script： " << error;  // Printing error message
    }
}

void GlobalView::run_globalVerify()
{
    if(!got_result_)
    {
        QMessageBox::warning(this, "Warning", "The computation of the Lipschitz constant "
                                              "for the three circuits is not finished; "
                                              "you have to wait for the computation or "
                                              "recalculate.");
        return;
    }

    bool flag = getVerifResultFromCsvfile(csvfile_, epsilon_, delta_);
    if(flag)  // The current model has been verified before.
    {
        QMessageBox::information(this, "Notice",
                                 QString("The file %1 has contained the (%2, %3)-robustness verification "
                                         "result for the three circuits, and it's now shown in the table").arg
                                 (csvfile_, QString::number(epsilon_), QString::number(delta_)));
    }
    else{
        QList<double> kList;
        kList << origin_lipschitz_ << random_lipschitz_ << specified_lipschitz_;
        global_verify(epsilon_, delta_, kList);
    }
}

void GlobalView::on_read_from_terminal_verif()
{
    while (process_veri_->bytesAvailable() > 0){
        QString line = process_veri_->readLine();
        output_.append(line);
        ui->textBrowser_output->append(line.simplified());
        if(line.startsWith("This model is"))
        {
            QString if_robust = "YES";
            if(line.startsWith("This model is not"))
            {
                if_robust = "NO";
            }
            qDebug() << line;
            qDebug() << if_robust;
            qDebug() << verif_count_;
            insertDataToTable(verif_count_, 1,
                                 QString("(%1, %2)").arg(QString::number(epsilon_),
                                                         QString::number(delta_)));
            insertDataToTable(verif_count_, 4, if_robust);
            verif_count_++;
        }
        else if(line.contains("The Global Verification End") && verif_count_%3 == 0){
            ui->table_res->setSpan(verif_count_-3, 1, 3, 1);  // merge the third column
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
    ui->doubleSpinBox_epsilon->setValue(ui->slider_epsilon->value()* 0.001);
    epsilon_ = ui->doubleSpinBox_epsilon->value();
    qDebug() << "epsilon: " << epsilon_;
}

void GlobalView::on_doubleSpinBox_epsilon_valueChanged(double pos)
{
    ui->slider_epsilon->setValue(ui->doubleSpinBox_epsilon->value()/ 0.001);
    epsilon_ = ui->doubleSpinBox_epsilon->value();
}

void GlobalView::on_slider_delta_sliderMoved(int pos)
{
    ui->doubleSpinBox_delta->setValue(ui->slider_delta->value()* 0.001);
    delta_ = ui->doubleSpinBox_delta->value();
    qDebug() << "delta: " << delta_;
}

void GlobalView::on_doubleSpinBox_delta_valueChanged(double pos)
{
    ui->slider_delta->setValue(ui->doubleSpinBox_delta->value()/ 0.001);
    delta_ = ui->doubleSpinBox_delta->value();
}

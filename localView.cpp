#include "localView.h"
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


LocalView::LocalView(QWidget *parent):
    QWidget(parent),
    ui(new Ui::localView)
{
    qputenv("QTWEBENGINE_REMOTE_DEBUGGING", "7777");
    ui->setupUi(this);
    this->init();

    connect(ui->radioButton_importfile, SIGNAL(pressed()), this, SLOT(on_radioButton_importfile_clicked()));
    connect(ui->pushButton_importdata, SIGNAL(pressed()), this, SLOT(importData()));
    connect(ui->radioButton_qubit, SIGNAL(pressed()), this, SLOT(on_radioButton_qubit_clicked()));
    connect(ui->radioButton_phaseRecog, SIGNAL(pressed()), this, SLOT(on_radioButton_phaseRecog_clicked()));
    connect(ui->radioButton_excitation, SIGNAL(pressed()), this, SLOT(on_radioButton_excitation_clicked()));
    connect(ui->radioButton_mnist, SIGNAL(pressed()), this, SLOT(on_radioButton_mnist_clicked()));
    connect(ui->radioButton_pure, SIGNAL(toggled(bool)), this, SLOT(on_radioButton_pure_clicked()));
    connect(ui->radioButton_mixed, SIGNAL(toggled(bool)), this, SLOT(on_radioButton_mixed_clicked()));
    connect(ui->checkBox_show_AE, SIGNAL(on_process_stateChanged(int)), this, SLOT(on_checkBox_show_AE_stateChanged(int)));
    connect(ui->checkBox_get_newdata, SIGNAL(on_process_stateChanged(int)), this, SLOT(on_checkBox_get_newdata_stateChanged(int)));
    connect(ui->slider_unit, SIGNAL(valueChanged(int)), this, SLOT(on_slider_unit_sliderMoved(int)));
    connect(ui->spinBox_unit, SIGNAL(valueChanged(int)), this, SLOT(on_spinBox_unit_valueChanged(int)));
    connect(ui->slider_batchnum, SIGNAL(valueChanged(int)), this, SLOT(on_slider_batchnum_sliderMoved(int)));
    connect(ui->spinBox_batchnum, SIGNAL(valueChanged(int)), this, SLOT(on_spinBox_batchnum_valueChanged(int)));
    connect(ui->radioButton_bitflip, SIGNAL(pressed()), this, SLOT(on_radioButton_bitflip_clicked()));
    connect(ui->radioButton_depolarizing, SIGNAL(pressed()), this, SLOT(on_radioButton_depolarizing_clicked()));
    connect(ui->radioButton_phaseflip, SIGNAL(pressed()), this, SLOT(on_radioButton_phaseflip_clicked()));
    connect(ui->radioButton_mixednoise, SIGNAL(pressed()), this, SLOT(on_radioButton_mixednoise_clicked()));
    connect(ui->radioButton_custom_noise, SIGNAL(pressed()), this, SLOT(on_radioButton_importkraus_clicked()));
    connect(ui->slider_prob, SIGNAL(valueChanged(int)), this, SLOT(on_slider_prob_sliderMoved(int)));
    connect(ui->doubleSpinBox_prob, SIGNAL(valueChanged(double)), this, SLOT(on_doubleSpinBox_prob_valueChanged(double)));
    connect(ui->pushButton_run, SIGNAL(pressed()), this, SLOT(run_localVeri()));
    connect(ui->pushButton_stop, SIGNAL(pressed()), this, SLOT(stopProcess()));
}

void LocalView::on_radioButton_bitflip_clicked()
{
    noise_type_ = "bit_flip";

    kraus_file_.fileName().clear();
    ui->lineEdit_custom_noise->clear();
    ui->radioButton_custom_noise->setChecked(false);
}

void LocalView::on_radioButton_depolarizing_clicked()
{
    noise_type_ = "depolarizing";

    kraus_file_.fileName().clear();
    ui->lineEdit_custom_noise->clear();
    ui->radioButton_custom_noise->setChecked(false);
}

void LocalView::on_radioButton_phaseflip_clicked()
{
    noise_type_ = "phase_flip";

    kraus_file_.fileName().clear();
    ui->lineEdit_custom_noise->clear();
    ui->radioButton_custom_noise->setChecked(false);
}

void LocalView::on_radioButton_mixednoise_clicked()
{
    noise_type_ = "mixed";

    kraus_file_.fileName().clear();
    ui->lineEdit_custom_noise->clear();
    ui->radioButton_custom_noise->setChecked(false);
}

void LocalView::on_radioButton_importkraus_clicked()
{
    noise_type_ = "custom";
    QString fileName = QFileDialog::getOpenFileName(this, "Open file", localDir+"/kraus");
    QFile file(fileName);
    kraus_file_ = QFileInfo(fileName);

    // model_name_ = fileName.mid(fileName.lastIndexOf("/")+1, fileName.indexOf(".")-fileName.lastIndexOf("/")-1);
    // qDebug() << "model_name_: " << model_name_;

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

void LocalView::on_slider_prob_sliderMoved(int pos)
{
    noise_prob_ = ui->slider_prob->value() * 0.00001;
    ui->doubleSpinBox_prob->setValue(noise_prob_);
    qDebug() << "noise_prob: " << noise_prob_;
}

void LocalView::on_doubleSpinBox_prob_valueChanged(double pos)
{
    noise_prob_ = ui->doubleSpinBox_prob->value();
    ui->slider_prob->setValue(noise_prob_ / 0.00001);
}

void LocalView::init()
{
    QString path = QApplication::applicationFilePath();
    if(path.contains("/build-VeriQR"))
    {
        localDir = path.mid(0, path.indexOf("/build-VeriQR")) + "/VeriQR/py_module/Local";
    }
    else if(path.contains("VeriQR/build/"))
    {
        localDir = path.mid(0, path.indexOf("/build")) + "/py_module/Local";
    }
    qDebug() << path;
    qDebug() << "localDir: " << localDir;
    // qDebug() << "noise: " << noise_type_;

    ui->textBrowser_output->setGeometry(QRect(20, 10, 67, 17));

    // Set the background color to transparent
    QPalette palette = ui->groupBox_run->palette();
    palette.setBrush(QPalette::Window, Qt::transparent);
    ui->groupBox_run->setPalette(palette);

    palette = ui->groupBox_runtime->palette();
    palette.setBrush(QPalette::Window, Qt::transparent);
    ui->groupBox_runtime->setPalette(palette);

    // MultiSelectComboBox for mnist digits
    comboBox_digits = new MultiSelectComboBox(ui->groupBox_file);
    comboBox_digits->setObjectName("comboBox_digits");
    ui->gridLayout->addWidget(comboBox_digits, 1, 2, 1, 1);
    QStringList digitsList;
    digitsList << "0" << "1" << "2" << "3" << "4" << "5" << "6" << "7" << "8" << "9";
    comboBox_digits->setMaxSelectNum(2);
    comboBox_digits->addItems_for_mnist(digitsList);

    // MultiSelectComboBox for noise
    QSizePolicy sizePolicy1(QSizePolicy::Expanding, QSizePolicy::Fixed);
    sizePolicy1.setHorizontalStretch(0);
    sizePolicy1.setVerticalStretch(0);
    sizePolicy1.setHeightForWidth(ui->pushButton_importdata->sizePolicy().hasHeightForWidth());

    comboBox_mixednoise = new MultiSelectComboBox(ui->groupBox_noisetype);
    comboBox_mixednoise->setObjectName("comboBox_mixednoise");
    sizePolicy1.setHeightForWidth(comboBox_mixednoise->sizePolicy().hasHeightForWidth());
    comboBox_mixednoise->setSizePolicy(sizePolicy1);
    ui->gridLayout_2->addWidget(comboBox_mixednoise, 0, 5, 1, 1);
    QStringList noiseList;
    noiseList << "bit flip" << "depolarizing" << "phase flip";
    comboBox_mixednoise->setMaxSelectNum(3);
    comboBox_mixednoise->addItems_for_noise(noiseList);
}

void LocalView::resizeEvent(QResizeEvent *)
{
    // if(showed_svg)
    // {
    //     svgWidget->load(robustDir + "/figures/"+ model_name_ + "_model.svg");
    // }

    if(showed_AE_)
    {
        QString img_file = localDir + "/adversary_examples/";

        int i = 0;
        QObjectList list = ui->scrollAreaWidgetContents_AE->children();
        foreach(QObject *obj, list)
        {
            if(obj->inherits("QLabel"))
            {
                QLabel *imageLabel = qobject_cast<QLabel*>(obj);
                QImage image(img_file + adv_examples_[i]);
                QPixmap pixmap = QPixmap::fromImage(image);
                imageLabel->setPixmap(pixmap.scaledToHeight(
                    ui->tab_ad->height()*0.32, Qt::SmoothTransformation));
                // qDebug() << imageLabel->objectName();
                i++;
            }
        }
    }
}

void LocalView::clear_output()
{
    // Clear the window interface, including the table and pictures.
    output_line_.clear();
    output_.clear();
    ui->textBrowser_output->clear();

    csvfile_.clear();
    res_model = new QStandardItemModel();

    close_circuit_diagram();
    showed_svg = false;
    showed_pdf = false;

    delete_all_adversarial_examples();
    showed_AE_ = false;
    adv_examples_.clear();
}


/* Clear the window and reset all settings. */
void LocalView::reset_all()
{
    clear_output();

    // Reset the options in the interface and related variables.
    // Basic settings:
    npzfile_.clear();
    model_file_ = QFileInfo();
    data_file_ = QFileInfo();
    model_name_.clear();
    filename_.clear();
    ui->radioButton_qubit->setChecked(0);
    ui->radioButton_phaseRecog->setChecked(0);
    ui->radioButton_excitation->setChecked(0);
    ui->radioButton_mnist->setChecked(0);
    ui->radioButton_importfile->setChecked(0);
    ui->pushButton_importdata->setChecked(0);
    ui->lineEdit_modelfile->clear();
    ui->lineEdit_datafile->clear();
    comboBox_digits->line_edit_->clear();
    for(int i = 0; i < comboBox_digits->list_widget_->count(); i++)
    {
        QCheckBox *digits_check_box = static_cast<QCheckBox*>(
            comboBox_digits->list_widget_->itemWidget(
                comboBox_digits->list_widget_->item(i)
            )
        );
        digits_check_box->setChecked(0);
    }

    state_type_ = "mixed";
    ui->radioButton_mixed->setChecked(1);

    need_to_visualize_AE_ = false;
    need_new_dataset_ = false;
    ui->checkBox_show_AE->setChecked(0);
    ui->checkBox_get_newdata->setChecked(0);

    robustness_unit_ = 1e-5;
    ui->spinBox_unit->setValue(5);
    ui->slider_unit->setValue(5);
    bacth_num_ = 5;
    ui->spinBox_batchnum->setValue(bacth_num_);
    ui->slider_batchnum->setValue(bacth_num_);

    noise_type_.clear();
    ui->radioButton_bitflip->setChecked(0);
    ui->radioButton_depolarizing->setChecked(0);
    ui->radioButton_phaseflip->setChecked(0);
    ui->radioButton_mixed->setChecked(0);
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
}

/* Open a txt file that records the runtime output. */
void LocalView::openFile(){
    QString filename = QFileDialog::getOpenFileName(this, "Open file", localDir+"/results/runtime_output");
    QFile file(filename);

    if (!file.open(QIODevice::ReadOnly |QIODevice::Text)) {
        QMessageBox::warning(this, "Warning", "Unable to open the file: " + file.errorString());
        return;
    }else if(QFileInfo(filename).suffix() != "txt") {
        QMessageBox::warning(this, "Warning", "VeriQR only supports .txt result data files.");
        return;
    }

    reset_all();

    output_ = QString::fromLocal8Bit(file.readAll());
    ui->textBrowser_output->setText(output_);
    file.close();

    filename_ = QFileInfo(file).fileName();
    filename_.chop(4);
    qDebug() << filename_;
    // "qubit_0.001×5_mixed" or "fashion8_0.001×3_mixed_0.1_PhaseFlip"

    csvfile_ = localDir + "/results/result_tables/" + filename_ + ".csv";

    // Obtain parameters from the filename
    QStringList args = filename_.split("_");
    model_name_ = args[0];
    QString unit = args[1].mid(0, args[1].indexOf("×"));
    robustness_unit_ = unit.toDouble();
    bacth_num_ = args[1].mid(args[1].indexOf("×")+1).toInt();
    state_type_ = args[2];
    qDebug() << "model_name_: " << model_name_;

    if(case_list_.indexOf(model_name_) != -1)  // three original case
    {
        // show_circuit_diagram_pdf(localDir + "/figures/" + model_name_ + "_model.pdf");
        show_circuit_diagram_pdf();
    }
    else  // iris_0.001×3_mixed_BitFlip_0.13462  or  mnist01_0.001×5_pure_PhaseFlip_0.001
    {
        noise_prob_ = args[args.size()-1].toDouble();
        int noise_type_index = 3;
        noise_type_ = args[noise_type_index];

        // Noise settings change
        if(noise_type_ == "PhaseFlip"){
            ui->radioButton_phaseflip->setChecked(1);
        }
        else if(noise_type_ == "BitFlip"){
            ui->radioButton_bitflip->setChecked(1);
        }
        else if(noise_type_ == "Depolarizing"){
            ui->radioButton_depolarizing->setChecked(1);
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
        else if(noise_type_ == "custom")  // iris_0.001×3_mixed_custom_kraus1qubit_0.001
        {
            kraus_file_ = QFileInfo(localDir + "/kraus/" + args[4] + ".npz");
            ui->lineEdit_custom_noise->setText(kraus_file_.filePath());
        }

        ui->doubleSpinBox_prob->setValue(noise_prob_);
        qDebug() << ui->slider_prob->value();

        if (file.open(QIODevice::ReadOnly | QIODevice::Text))
        {
            while (!file.atEnd())
            {
                QByteArray line = file.readLine();
                QString str(line);
                // qDebug() << str;
                // displayString << str;
                if(str.contains(".svg was saved"))
                {
                    show_circuit_diagram_svg(localDir + "/figures/" +
                                             str.mid(0, str.indexOf(".svg was saved")+4));
                }
            }
            file.close();
        }

        // for (QString output_line_: output_)
        // {
        //     if(output_line_.contains(".svg was saved"))
        //     {
        //         show_circuit_diagram_svg(localDir + "/figures/" +
        //                                  output_line_.mid(0, output_line_.indexOf(".svg was saved")+4));
        //     }
        // }
        // QString img_file = QString("%1/figures/%2_%3.svg").arg(
        //     localDir, model_name_, filename_.mid(filename_.indexOf(noise_type_)));
        // // show circuit diagram
        // show_circuit_diagram_svg(img_file);
    }

    // Selected model change
    if(model_name_ == "qubit"){
        ui->radioButton_qubit->setChecked(1);
    }
    else if(model_name_ == "phaseRecog"){
        ui->radioButton_phaseRecog->setChecked(1);
    }
    else if(model_name_ == "excitation"){
        ui->radioButton_excitation->setChecked(1);
    }
    else if(model_name_.contains("mnist")){
        ui->radioButton_mnist->setChecked(1);
        comboBox_digits->line_edit_->setText(QString(model_name_[5])+";"+QString(model_name_[6])+";");
        // qDebug() << comboBox_digits->current_select_items();
    }
    else{
        ui->radioButton_importfile->setChecked(1);
    }
    QString model_filename = localDir + "/model_and_data/" + model_name_ + ".qasm";
    model_file_ = QFileInfo(model_filename);
    ui->lineEdit_modelfile->setText(model_filename);

    QString data_filename = localDir + "/model_and_data/" + model_name_ + "_data.npz";
    data_file_ = QFileInfo(data_filename);
    ui->lineEdit_datafile->setText(data_filename);

    // Data type change
    if(state_type_ == "pure"){
        ui->radioButton_pure->setChecked(1);
    }
    else if(state_type_ == "mixed"){
        ui->radioButton_mixed->setChecked(1);
    }

    // Choice about adversarial examples change
    if(model_name_.contains("mnist") && state_type_ == "pure"){
        ui->checkBox_show_AE->setChecked(1);
        show_adversarial_examples();
    }
    else{
        ui->checkBox_show_AE->setChecked(0);
    }
    qDebug() << "visualizing_AE_: " << need_to_visualize_AE_;
    // qDebug() << "need_new_dataset_: " << need_new_dataset_;

    // Experiment settings change
    ui->slider_unit->setValue(unit.length() - 2);
    ui->spinBox_unit->setValue(unit.length() - 2);
    ui->slider_batchnum->setValue(bacth_num_);
    ui->spinBox_batchnum->setValue(bacth_num_);

    // Results visualization
    show_result_tables();

    get_table_data("openfile");
}

/* Save the runtime output as a txt file to the specified location. */
void LocalView::saveFile()
{
    if(ui->textBrowser_output->toPlainText().isEmpty()){
        QMessageBox::warning(this, "Warning", "No program was ever run and no results can be saved.");
        return;
    }

    output_ = ui->textBrowser_output->toPlainText();

    QString runtime_path = localDir + "/results/runtime_output/" + filename_ + ".txt";
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

void LocalView::saveasFile()
{
    QString fileName = QFileDialog::getSaveFileName(this, "Save as", localDir + "/results/runtime_output/");
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

/* Import an OpenQASM format file, representing the quantum circuit for a model. */
void LocalView::importModel()
{
    QString filename = QFileDialog::getOpenFileName(this, "Open file", localDir+"/model_and_data");
    QFile file(filename);
    model_file_ = QFileInfo(filename);

    model_name_ = filename.mid(filename.lastIndexOf("/")+1, filename.indexOf(".qasm"));
    qDebug() << "model_name_: " << model_name_;

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

void LocalView::on_radioButton_importfile_clicked(){
    if(ui->radioButton_importfile->isChecked()){
        importModel();
    }
    npzfile_.clear();
}

/* Import a '.npz' format file, encapsulating a quantum circuit,
 * a quantum measurement, and a training dataset. */
void LocalView::importData(){
    if(!ui->radioButton_importfile->isChecked()){
        QMessageBox::warning(this, "Warning", "Please select the model first! ");
        return;
    }

    QString filename = QFileDialog::getOpenFileName(this, "Open file", localDir+"/model_and_data");
    QFile file(filename);
    data_file_ = QFileInfo(filename);

    if (!file.open(QIODevice::ReadOnly)) {
        QMessageBox::warning(this, "Warning", "Unable to open the file: " + file.errorString());
        return;
    }else if(data_file_.suffix() != "npz"){
        QMessageBox::warning(this, "Warning", "VeriQR only supports .npz data files.");
        return;
    }

    ui->lineEdit_datafile->setText(data_file_.filePath());

    file.close();
}

void LocalView::on_radioButton_qubit_clicked()
{
    npzfile_ = "qubit.npz";
    model_name_ = npzfile_.mid(0, npzfile_.indexOf(".npz"));
    qDebug() << npzfile_;

    model_file_.fileName().clear();
    data_file_.fileName().clear();
    ui->lineEdit_modelfile->clear();
    ui->lineEdit_datafile->clear();

    if(ui->checkBox_show_AE->isChecked()){
        ui->checkBox_show_AE->setChecked(0);  // 取消选中
    }
}

void LocalView::on_radioButton_phaseRecog_clicked()
{
    npzfile_ = "phaseRecog.npz";
    model_name_ = npzfile_.mid(0, npzfile_.indexOf(".npz"));
    qDebug() << npzfile_;

    model_file_.fileName().clear();
    data_file_.fileName().clear();
    ui->lineEdit_modelfile->clear();
    ui->lineEdit_datafile->clear();

    if(ui->checkBox_show_AE->isChecked()){
        ui->checkBox_show_AE->setChecked(0);  // 取消选中
    }
}

void LocalView::on_radioButton_excitation_clicked()
{
    npzfile_ = "excitation.npz";
    model_name_ = npzfile_.mid(0, npzfile_.indexOf(".npz"));
    qDebug() << npzfile_;

    model_file_.fileName().clear();
    data_file_.fileName().clear();
    ui->lineEdit_modelfile->clear();
    ui->lineEdit_datafile->clear();

    if(ui->checkBox_show_AE->isChecked()){
        ui->checkBox_show_AE->setChecked(0);  // 取消选中
    }
}

void LocalView::on_radioButton_mnist_clicked()
{
    npzfile_ = "mnist.npz";
    model_name_ = npzfile_.mid(0, npzfile_.indexOf(".npz"));
    qDebug() << npzfile_;

    model_file_.fileName().clear();
    data_file_.fileName().clear();
    ui->lineEdit_modelfile->clear();
    ui->lineEdit_datafile->clear();
}

void LocalView::on_radioButton_pure_clicked()
{
    state_type_ = "pure";
    qDebug() << state_type_;
}

void LocalView::on_radioButton_mixed_clicked()
{
    state_type_ = "mixed";
    qDebug() << state_type_;

    if(ui->checkBox_show_AE->isChecked()){
        ui->checkBox_show_AE->setChecked(0);  // Uncheck
    }
}

void LocalView::on_checkBox_show_AE_stateChanged(int state)
{
    if(state == Qt::Checked)
    {
        if(!model_name_.contains("mnist") or state_type_ != "pure")
        {
            need_to_visualize_AE_ = false;
            ui->checkBox_show_AE->setChecked(0);  // reset
            QMessageBox::warning(this, "Warning",
                                 "Only the adversarial examples generated by "
                                 "the mnist model can be visualized. ");
        }
        else
        {
            need_to_visualize_AE_ = true;
        }
    }
    else  // Qt::Unchecked
    {
        need_to_visualize_AE_ = false;
    }
    // qDebug() << ui->checkBox->isChecked();
}


void LocalView::on_checkBox_get_newdata_stateChanged(int state)
{
    if (state == Qt::Checked)
    {
        need_new_dataset_ = true;
    }
    else
    {
        need_new_dataset_ = false;
    }
}

void LocalView::on_slider_unit_sliderMoved(int pos)
{
    ui->spinBox_unit->setValue(pos);
    //    qDebug() << robustness_unit_;
}

void LocalView::on_spinBox_unit_valueChanged(int pos)
{
    ui->slider_unit->setValue(pos);
    //    qDebug() << robustness_unit_;
}

void LocalView::on_slider_batchnum_sliderMoved(int pos)
{
    bacth_num_ = pos;
    ui->spinBox_batchnum->setValue(pos);
}

void LocalView::on_spinBox_batchnum_valueChanged(int pos)
{
    bacth_num_ = pos;
    ui->slider_batchnum->setValue(pos);
}

void LocalView::run_localVeri()
{
    if((model_file_.fileName().isEmpty() || data_file_.fileName().isEmpty())  // No model was selected
        && npzfile_.isEmpty())
    {
        QMessageBox::warning(this, "Warning", "You should choose a model! ");
        return;
    }

    clear_output();

    robustness_unit_ = pow(0.1, ui->slider_unit->value());
    bacth_num_ = ui->slider_batchnum->value();
    QString unit = QString::number(robustness_unit_);
    QString batch_num = QString::number(bacth_num_);

    QString cmd = "python";
    QString excuteFile = "local_verif.py";
    QString paramsList;
    QStringList args;
    // qDebug() << batchnum;

    if(!npzfile_.isEmpty())
    {
        if(npzfile_.contains("mnist"))
        {
            QString digits = comboBox_digits->current_select_items().join("");
            QString qasmfile = QString("./model_and_data/mnist%1.qasm").arg(digits);
            QString datafile = QString("./model_and_data/mnist%1_data.npz").arg(digits);
            model_name_ = QString("mnist%1").arg(digits);
            args << excuteFile << qasmfile << datafile << unit << batch_num << state_type_;
        }
        else
        {
            QString npzfile = "./model_and_data/" + npzfile_; // npzfile is complete path
            args << excuteFile << npzfile << unit << batch_num << state_type_;
        }
    }
    else    // has imported a file
    {
        QString qasmfile = model_file_.filePath();
        QString datafile = data_file_.filePath();
        // model_name_ =
        args << excuteFile << qasmfile << datafile << unit << batch_num << state_type_;
    }

    // model_name be like: 'qubit'
    if(model_name_.contains("mnist"))
    {
        if(state_type_ == "pure"){
            ui->checkBox_show_AE->setChecked(1);
            args << "true";
        }
        else{
            ui->checkBox_show_AE->setChecked(0);
            args << "false";
        }
    }
    else
    {
        args << "false";
    }

    if(noise_type_ == "mixed")
    {
        args << noise_type_;
        mixed_noises_ = comboBox_mixednoise->current_select_items();
        for(int i = 0; i < mixed_noises_.count(); i++)
        {
            mixed_noises_[i] = mixed_noises_[i].replace(" ", "_");
            args << mixed_noises_[i];
        }
        args << QString::number(noise_prob_);
    }
    else if(noise_type_ == "custom")
    {
        QString krausfile = kraus_file_.filePath();
        args << noise_type_ << krausfile << QString::number(noise_prob_);
    }
    else if(!noise_type_.isEmpty())
    {
        args << noise_type_ << QString::number(noise_prob_);
    }

    paramsList = cmd + " " + args.join(" ");
    qDebug() << paramsList;

    show_result_tables();

    process = new QProcess(this);
    process->setReadChannel(QProcess::StandardOutput);
    connect(process, SIGNAL(stateChanged(QProcess::ProcessState)), SLOT(on_process_stateChanged(QProcess::ProcessState)));
    connect(process, SIGNAL(readyReadStandardOutput()), this, SLOT(on_read_output()));

    process->setWorkingDirectory(localDir);
    process->start(cmd, args);
    if(!process->waitForStarted())
    {
        qDebug() << "Process failure! Error: " << process->errorString();
    }
    else
    {
        qDebug() << "Process succeed! ";
    }

    if (!process->waitForFinished()){
        qDebug() << "wait";
        QCoreApplication::processEvents(QEventLoop::AllEvents, 2000);
    }

    QString error = process->readAllStandardError();  // Command line error message
    if(!error.isEmpty()){
        qDebug()<< "Error executing script： " << error;  // Printing error message
    }
}

void LocalView::on_process_stateChanged(QProcess::ProcessState state)
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

void LocalView::stopProcess()
{
    this->process->terminate();
    this->process->waitForFinished();
    qDebug() << "Process terminate!";
}

void LocalView::on_read_output()
{
    // bool is_case = circuit_diagram_map.find(model_name_) != circuit_diagram_map.end();
    bool is_case = (case_list_.indexOf(model_name_) != -1);
    while (process->bytesAvailable() > 0){
        output_line_ = process->readLine();
        // qDebug() << output_line_;
        output_.append(output_line_);
        ui->textBrowser_output->append(output_line_.simplified());

        if(is_case && output_line_.contains("Starting") && !showed_pdf)
        {
            // show_circuit_diagram_pdf(localDir + "/figures/" + model_name_ + "_model.pdf");
            show_circuit_diagram_pdf();
        }
        // else if(!is_case && output_line_.contains(".svg was saved") && !showed_svg)
        else if(!is_case && output_line_.contains(".svg was saved"))
        {
            show_circuit_diagram_svg(localDir + "/figures/" +
                                     output_line_.mid(0, output_line_.indexOf(".svg was saved")+4));
        }

        // Verification over, show results and adversarial examples.
        else if(output_line_.contains(".csv was saved!"))
        {
            csvfile_ = output_line_.mid(0, output_line_.indexOf(".csv was saved")+4);
            filename_ = csvfile_.mid(0, csvfile_.indexOf(".csv"));
            csvfile_ = localDir + "/results/result_tables/" + csvfile_;

            get_table_data("run");

            if(ui->checkBox_show_AE->isChecked())
            {
                show_adversarial_examples();
            }
            break;
        }
    }
}

void LocalView::show_adversarial_examples()
{
    QString img_file = localDir + "/adversary_examples";
    // qDebug() << img_file;

    QDir dir(img_file);
    QStringList img_names;

    if (!dir.exists()) img_names = QStringList("");

    dir.setFilter(QDir::Files);
    dir.setSorting(QDir::Name);

    img_names = dir.entryList();
    // qDebug() << dir.entryList();
    QString digits = comboBox_digits->current_select_items().join("");
    digits = "advExample_" + digits;
    qDebug() << "The selected numbers are: " << digits;
    for (int i = 0; i < img_names.size(); ++i)
    {
        if(img_names[i].startsWith(digits))
        {
            QLabel *imageLabel_AE;
            imageLabel_AE= new QLabel(ui->scrollAreaWidgetContents_AE);
            imageLabel_AE->setObjectName(QString::fromUtf8("imageLabel_AE_")+QString::number(i));
            imageLabel_AE->setAlignment(Qt::AlignCenter);

            QImage image(img_file + "/" + img_names[i]);
            QPixmap pixmap = QPixmap::fromImage(image);
            imageLabel_AE->setPixmap(pixmap);

            ui->verticalLayout_AE->addWidget(imageLabel_AE);
            adv_examples_.append(img_names[i]);
        }
    }
    showed_AE_ = true;
}

void LocalView::delete_all_adversarial_examples()
{
    // Nothing needs to be done when no adversarial examples are shown.
    if(!showed_AE_) return;

    QLayoutItem *child;
    int i = 0;
    while((child = ui->verticalLayout_AE->takeAt(i)) != nullptr)
    {
        if(child->widget())
        {
            child->widget()->setParent(nullptr);
            qDebug() << "delete " << child->widget()->objectName() << "!";
            ui->verticalLayout_AE->removeWidget(child->widget());
            delete child->widget();
        }
    }
    showed_AE_ = false;
    adv_examples_.clear();
    qDebug() << "delete all adversarial examples! ";
}

void LocalView::show_circuit_diagram_svg(QString img_file)
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

        // QScrollArea *scrollArea_origin_circ = new QScrollArea(ui->tab_circ);
        // scrollArea_origin_circ->setObjectName("scrollArea_origin_circ");
        // scrollArea_origin_circ->setWidgetResizable(true);
        // QWidget *scrollAreaWidgetContents_origin_circ = new QWidget();
        // scrollAreaWidgetContents_origin_circ->setObjectName("scrollAreaWidgetContents_origin_circ");
        // scrollArea_origin_circ->setWidget(scrollAreaWidgetContents_origin_circ);
        // ui->verticalLayout_circ->addWidget(scrollArea_origin_circ);
        // SvgWidget *svgWidget = new SvgWidget(scrollAreaWidgetContents_origin_circ);
        // svgWidget->load(img_file);
        // svgWidget->setObjectName("svgWidget_origin");
        // QVBoxLayout *verticalLayout_origin = new QVBoxLayout(scrollAreaWidgetContents_origin_circ);
        // verticalLayout_origin->setObjectName("verticalLayout_origin");
        // verticalLayout_origin->addWidget(svgWidget);
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

        // QScrollArea *scrollArea_random_circ = new QScrollArea(ui->tab_circ);
        // scrollArea_random_circ->setObjectName("scrollArea_random_circ");
        // scrollArea_random_circ->setWidgetResizable(true);
        // QWidget *scrollAreaWidgetContents_random_circ = new QWidget();
        // scrollAreaWidgetContents_random_circ->setObjectName("scrollAreaWidgetContents_random_circ");
        // scrollArea_random_circ->setWidget(scrollAreaWidgetContents_random_circ);
        // ui->verticalLayout_circ->addWidget(scrollArea_random_circ);
        // SvgWidget *svgWidget = new SvgWidget(scrollAreaWidgetContents_random_circ);
        // svgWidget->load(img_file);
        // svgWidget->setObjectName("svgWidget_random");
        // QVBoxLayout *verticalLayout_random = new QVBoxLayout(scrollAreaWidgetContents_random_circ);
        // verticalLayout_random->setObjectName("verticalLayout_random");
        // verticalLayout_random->addWidget(svgWidget);
    }
    else
    {
        QGroupBox *groupBox_final_circ = new QGroupBox(ui->scrollArea_circ);
        groupBox_final_circ->setObjectName("groupBox_random_circ");
        groupBox_final_circ->setTitle("Circuit with specified noise");
        groupBox_final_circ->setFont(font);
        ui->verticalLayout_circ->addWidget(groupBox_final_circ, 1);
        QVBoxLayout *verticalLayout_final = new QVBoxLayout(groupBox_final_circ);
        verticalLayout_final->setObjectName("verticalLayout_final");

        SvgWidget *svgWidget = new SvgWidget(groupBox_final_circ);
        svgWidget->load(img_file);
        svgWidget->setObjectName("svgWidget_final");
        verticalLayout_final->addWidget(svgWidget);

        // QScrollArea *scrollArea_final_circ = new QScrollArea(ui->tab_circ);
        // scrollArea_final_circ->setObjectName("scrollArea_final_circ");
        // scrollArea_final_circ->setWidgetResizable(true);
        // QWidget *scrollAreaWidgetContents_final_circ = new QWidget();
        // scrollAreaWidgetContents_final_circ->setObjectName("scrollAreaWidgetContents_final_circ");
        // scrollArea_final_circ->setWidget(scrollAreaWidgetContents_final_circ);
        // ui->verticalLayout_circ->addWidget(scrollArea_final_circ);
        // SvgWidget *svgWidget = new SvgWidget(scrollAreaWidgetContents_final_circ);
        // svgWidget->load(img_file);
        // svgWidget->setObjectName("svgWidget_final");
        // QVBoxLayout *verticalLayout_final = new QVBoxLayout(scrollAreaWidgetContents_final_circ);
        // verticalLayout_final->setObjectName("verticalLayout_final");
        // verticalLayout_final->addWidget(svgWidget);
    }
    // double container_w = double(ui->scrollArea_final->width());
    // double svg_w = double(svgWidget->renderer()->defaultSize().width());
    // double svg_h = double(svgWidget->renderer()->defaultSize().height());
    // double iris_w = 977.0;
    // double iris_h = 260.0;
    // svg_h = container_w / svg_w * svg_h;
    // iris_h = container_w / iris_w * iris_h;
    // qDebug() << svg_h;
    // iris.svg: (width, height) = (977, 260)
    // ui->verticalLayout_circ->addWidget(svgWidget);

    // QSpacerItem *verticalSpacer2 = new QSpacerItem(20, 40, QSizePolicy::Expanding, QSizePolicy::Expanding);
    // ui->verticalLayout_circ->addItem(verticalSpacer2);
    // int diff = double(svg_h*2)/double(iris_h) * 1000;
    // // qDebug() << diff;
    // ui->verticalLayout_circ->insertWidget(1, svgWidget, svg_h);
    // ui->verticalLayout_circ->setStretch(0, 1.5*1000);
    // ui->verticalLayout_circ->setStretch(2, 1.5*1000);
    showed_svg = true;
}

// void LocalView::show_circuit_diagram_svg(QString img_file)
// {
//     QFile file(img_file);
//     qDebug() << "show " << img_file;
//     if(!file.open(QIODevice::ReadOnly))
//     {
//         QMessageBox::warning(this, "Warning", "Failed to open the file: " + img_file);
//         return;
//     }

//     QSpacerItem *verticalSpacer1 = new QSpacerItem(20, 40, QSizePolicy::Expanding, QSizePolicy::Expanding);
//     ui->verticalLayout_circ->addItem(verticalSpacer1);

//     svgWidget = new SvgWidget(ui->scrollAreaWidgetContents_circ);
//     svgWidget->load(img_file);
//     svgWidget->setObjectName("svgWidget_circ");
//     double container_w = double(ui->scrollAreaWidgetContents_circ->width());
//     double svg_w = double(svgWidget->renderer()->defaultSize().width());
//     double svg_h = double(svgWidget->renderer()->defaultSize().height());
//     double iris_w = 977.0;
//     double iris_h = 260.0;
//     svg_h = container_w / svg_w * svg_h;
//     iris_h = container_w / iris_w * iris_h;
//     // qDebug() << svg_h;
//     // iris.svg: (width, height) = (977, 260)

//     QSpacerItem *verticalSpacer2 = new QSpacerItem(20, 40, QSizePolicy::Expanding, QSizePolicy::Expanding);
//     ui->verticalLayout_circ->addItem(verticalSpacer2);
//     int diff = double(svg_h*2)/double(iris_h) * 1000;
//     // qDebug() << diff;
//     ui->verticalLayout_circ->insertWidget(1, svgWidget, diff);
//     ui->verticalLayout_circ->setStretch(0, 1.5*1000);
//     ui->verticalLayout_circ->setStretch(2, 1.5*1000);
//     showed_svg = true;
// }

void LocalView::close_circuit_diagram_svg()
{
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
}

void LocalView::show_circuit_diagram_pdf()
{    
    if(model_name_.isEmpty())
    {
        QMessageBox::warning(this, "Warning", "The model " + model_name_ + " do not exist. ");
        return;
    }

    QString img_file = localDir + "/figures/" + model_name_ + "_model.pdf";
    qDebug() << "figure file: " << img_file;
    QFile file(img_file);
    if(!file.open(QIODevice::ReadOnly))
    {
        QMessageBox::warning(this, "Warning", "Failed to open the file: " + img_file);
        return;
    }

    pdfView = new PdfView(ui->scrollArea_circ);
    pdfView->loadDocument(img_file);
    pdfView->setObjectName("pdfView_circ");
    ui->verticalLayout_circ->addWidget(pdfView);
    showed_pdf = true;
}

void LocalView::close_circuit_diagram_pdf()
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

void LocalView::close_circuit_diagram()
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

void LocalView::show_result_tables(){
    int columnCount = bacth_num_;
    res_model = new QStandardItemModel();
    // Set the column header.
    res_model->setHorizontalHeaderLabels(QStringList() << "epsilon" << "Circuit" <<
                                         "Rough Verif RA(%)" << "Rough Verif VT(s)" <<
                                         "Accurate Verif RA(%)" << "Accurate Verif VT(s)");

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
    ui->table_res->setColumnWidth(0, ui->table_res->width()/8*0.8);
    ui->table_res->setColumnWidth(1, ui->table_res->width()/8*1.8);
    ui->table_res->setColumnWidth(2, ui->table_res->width()/8*1.3);
    ui->table_res->setColumnWidth(3, ui->table_res->width()/8*1.3);
    ui->table_res->setColumnWidth(4, ui->table_res->width()/8*1.4);
    ui->table_res->setColumnWidth(5, ui->table_res->width()/8*1.4);

    QString eps = QString::number(ui->slider_unit->value());
    // Initial table data.
    for(int row_index = 0; row_index < columnCount; row_index++)
    {
        // Set the row header as epsilon for each validation experiment.
        QString eps_str = QString::number(row_index + 1) + QString::fromLocal8Bit("e-") + eps;
        // For original circuit
        QStandardItem *item = new QStandardItem(eps_str);
        item->setTextAlignment(Qt::AlignCenter);
        res_model->setItem(row_index * 3, 0, item);
        // For circuit with random noise
        item = new QStandardItem(eps_str);
        item->setTextAlignment(Qt::AlignCenter);
        res_model->setItem(row_index * 3 + 1, 0, item);
        // For circuit with specified noise
        item = new QStandardItem(eps_str);
        item->setTextAlignment(Qt::AlignCenter);
        res_model->setItem(row_index * 3 + 2, 0, item);

        ui->table_res->setSpan(row_index * 3, 0, 3, 1);
        for(int col_index = 1; col_index < 6; col_index++)
        {
            item = new QStandardItem(QString("-"));
            item->setTextAlignment(Qt::AlignCenter);
            res_model->setItem(row_index, col_index, item);
        }
    }
}

/* Parse the result file contents into a table */
void LocalView::get_table_data(QString op)
{
    // When the program fails
    if(op == "run" && process->exitStatus() != QProcess::NormalExit)
    {
        qDebug() << process->exitStatus();
        QMessageBox::warning(this, "Warning", "Program abort.");
        return;
    }
    // When the program ends normally
    QFile file(csvfile_);
    qDebug() << "csvfile: " << csvfile_;
    if(!file.open(QIODevice::ReadOnly | QIODevice::Text)){
        QMessageBox::warning(this, "Warning", "Unable to open the file: "
                                                  + csvfile_ + "\n" + file.errorString());
        return;
    }

    QTextStream in(&file);
    int row_index = 0;
    while (!in.atEnd()) {
        QString line = in.readLine();
        if(row_index > 0){
            QStringList res_fields = line.split(",");
            QStandardItem *circuit_item = new QStandardItem(QString(res_fields[1]));
            QStandardItem *RA_1_item = new QStandardItem(QString(res_fields[2]));
            QStandardItem *VT_1_item = new QStandardItem(QString(res_fields[3]));
            QStandardItem *RA_2_item = new QStandardItem(QString(res_fields[4]));
            QStandardItem *VT_2_item = new QStandardItem(QString(res_fields[5]));
            circuit_item->setTextAlignment(Qt::AlignCenter);
            RA_1_item->setTextAlignment(Qt::AlignCenter);
            VT_1_item->setTextAlignment(Qt::AlignCenter);
            RA_2_item->setTextAlignment(Qt::AlignCenter);
            VT_2_item->setTextAlignment(Qt::AlignCenter);
            res_model->setItem(row_index-1, 1, circuit_item);
            res_model->setItem(row_index-1, 2, RA_1_item);
            res_model->setItem(row_index-1, 3, VT_1_item);
            res_model->setItem(row_index-1, 4, RA_2_item);
            res_model->setItem(row_index-1, 5, VT_2_item);
        }
        row_index++;
    }
    file.close();
}

LocalView::~LocalView()
{
    delete ui;
}

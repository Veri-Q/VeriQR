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
    connect(ui->pushButton_run, SIGNAL(pressed()), this, SLOT(run_localVeri()));
    connect(ui->pushButton_stop, SIGNAL(pressed()), this, SLOT(stopProcess()));
    // connect(ui->comboBox, SIGNAL(currentTextChanged(QString)), this, SLOT(on_comboBox_currentTextChanged(QString)));
    connect(ui->radioButton_bitflip, SIGNAL(pressed()), this, SLOT(on_radioButton_bitflip_clicked()));
    connect(ui->radioButton_depolarizing, SIGNAL(pressed()), this, SLOT(on_radioButton_depolarizing_clicked()));
    connect(ui->radioButton_phaseflip, SIGNAL(pressed()), this, SLOT(on_radioButton_phaseflip_clicked()));
    connect(ui->radioButton_mixednoise, SIGNAL(pressed()), this, SLOT(on_radioButton_mixednoise_clicked()));
    connect(ui->radioButton_custom_noise, SIGNAL(pressed()), this, SLOT(on_radioButton_importops_clicked()));
    connect(ui->slider_prob, SIGNAL(valueChanged(int)), this, SLOT(on_slider_prob_sliderMoved(int)));
    connect(ui->doubleSpinBox_prob, SIGNAL(valueChanged(double)), this, SLOT(on_doubleSpinBox_prob_valueChanged(double)));
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

void LocalView::on_slider_prob_sliderMoved(int pos)
{
    ui->doubleSpinBox_prob->setValue(ui->slider_prob->value()* 0.001);
    noise_prob_ = ui->doubleSpinBox_prob->value();
    qDebug() << "noise_prob: " << noise_prob_;
}

void LocalView::on_doubleSpinBox_prob_valueChanged(double pos)
{
    ui->slider_prob->setValue(ui->doubleSpinBox_prob->value()/ 0.001);
    noise_prob_ = ui->doubleSpinBox_prob->value();
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

    ui->textBrowser_output->setGeometry(QRect(20, 10, 67, 17));

    QPalette palette = ui->groupBox_run->palette();
    palette.setBrush(QPalette::Window, Qt::transparent);  // 设置背景颜色为透明
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

void LocalView::on_radioButton_importops_clicked()
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

void LocalView::resizeEvent(QResizeEvent *)
{
    // if(showed_svg)
    // {
    //     svgWidget->load(robustDir + "/Figures/"+ model_name_ + "_model.svg");
    // }

    if(showed_adexample)
    {
        QString img_file = localDir + "/adversary_examples/";

        int i = 0;
        QObjectList list = ui->scrollAreaWidgetContents_ad->children();
        foreach(QObject *obj, list)
        {
            if(obj->inherits("QLabel"))
            {
                QLabel *imageLabel = qobject_cast<QLabel*>(obj);
                QImage image(img_file + adv_examples[i]);
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
void LocalView::openFile(){
    QString fileName = QFileDialog::getOpenFileName(this, "Open file", localDir+"/results/runtime_output");
    QFile file(fileName);

    if (!file.open(QIODevice::ReadOnly |QIODevice::Text)) {
        QMessageBox::warning(this, "Warning", "Unable to open the file: " + file.errorString());
        return;
    }else if(QFileInfo(fileName).suffix() != "txt") {
        QMessageBox::warning(this, "Warning", "VeriQR only supports .txt result data files.");
        return;
    }

    close_circuit_diagram();
    delete_all_adversary_examples();

    output_ = QString::fromLocal8Bit(file.readAll());
    ui->textBrowser_output->setText(output_);

    file_name_ = QFileInfo(file).fileName();
    file_name_.chop(4);
    qDebug() << file_name_;
    // "binary_0.001_2_mixed" or "FashionMNIST_Ent1_0.001_3_mixed_0.1_PhaseFlip"

    csvfile_ = localDir + "/results/result_tables/" + file_name_ + ".csv";

    // 从文件名获取各种参数信息
    QStringList args = file_name_.split("_");
    QString unit, img_file;
    if(args.size() == 4){   // four original case
        model_name_ = args[0];
        unit = args[1];
        robustness_unit_ = unit.toDouble();
        experiment_number_ = args[2].toInt();
        state_type_ = args[3];
        img_file = model_name_;
        if(model_name_.startsWith("mnist") && model_name_.size() > 5){  // mnist17_0.001_1_pure
            show_circuit_diagram_svg(localDir+"/Figures/"+img_file+"_model.svg");
        }
        else  // binary_0.001_5_mixed or mnist_0.001_3_pure etc.
        {
            show_circuit_diagram_pdf(localDir+"/Figures/"+img_file+"_model.pdf");
        }
    }
    else{  // iris_0.001_3_mixed_0.13462_BitFlip
        unit = args[args.size()-5];
        robustness_unit_ = unit.toDouble();
        experiment_number_ = args[args.size()-4].toInt();
        state_type_ = args[args.size()-3];
        noise_prob_ = args[args.size()-2].toDouble();
        noise_type_ = args[args.size()-1];
        model_name_ = file_name_.mid(0, file_name_.indexOf(unit)-1);
        // show circuit diagram
        img_file = model_name_ + "_with_" + args[args.size()-2] + "_" + noise_type_;
        show_circuit_diagram_svg(localDir+"/Figures/"+img_file+"_model.svg");
        // model_file_ = QFileInfo(localDir+"/model_and_data/"+model_name_+".qasm");
        // data_file_ = QFileInfo(localDir+"/model_and_data/"+model_name_+"_data.npz");
        // ui->lineEdit_modelfile->setText(model_file_.filePath());
        // ui->lineEdit_datafile->setText(data_file_.filePath());
        model_change_to_ui();
    }
    qDebug() << "model_name_: " << model_name_;

    ui->slider_unit->setValue(unit.length() - 2);
    ui->spinBox_unit->setValue(unit.length() - 2);

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
    else if(model_name_.contains("mnist")){
        ui->radioButton_mnist->setChecked(1);
        if(model_name_.size() == 5)
        {
            comboBox_digits->setToolTip("3 & 6");
        }
        else
        {
            // QString text = QString(model_name_[5]) + " & " +QString(model_name_[6]);
            // ui->comboBox->setCurrentIndex(ui->comboBox->findText(text));
            comboBox_digits->setToolTip(QString(model_name_[5]) + " & " +QString(model_name_[6]));
        }
        qDebug() << comboBox_digits->current_select_items();
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

    if(model_name_.contains("mnist") && state_type_ == "pure"){
        ui->checkBox->setChecked(1);
        show_adversary_examples();
    }

    show_result_tables();

    get_table_data("openfile");

    file.close();
}

void LocalView::model_change_to_ui(){
    // selected model change
    ui->lineEdit_modelfile->setText(localDir+"/model_and_data/"+model_name_+".qasm");
    ui->lineEdit_datafile->setText(localDir+"/model_and_data/"+model_name_+"_data.npz");

    // noise type change
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
    }

    // noise probability change
    ui->doubleSpinBox_prob->setValue(noise_prob_);
    qDebug() << ui->slider_prob->value();
}


/* 将运行时输出信息存为txt文件 */
void LocalView::saveFile()
{
    if(ui->textBrowser_output->toPlainText().isEmpty()){
        QMessageBox::warning(this, "Warning", "No program was ever run and no results can be saved.");
        return;
    }

    output_ = ui->textBrowser_output->toPlainText();

    QString runtime_path = localDir + "/results/runtime_output/" + file_name_ + ".txt";
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

/* 导入.qasm数据文件 */
void LocalView::importModel()
{
    QString fileName = QFileDialog::getOpenFileName(this, "Open file", localDir+"/model_and_data");
    QFile file(fileName);
    model_file_ = QFileInfo(fileName);

    model_name_ = fileName.mid(fileName.lastIndexOf("/")+1, fileName.indexOf(".")-fileName.lastIndexOf("/")-1);
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
    npzfile_ = "";
}

/* 导入.npz数据文件 */
void LocalView::importData(){
    if(!ui->radioButton_importfile->isChecked()){
        QMessageBox::warning(this, "Warning", "Please select the model first! ");
        return;
    }

    QString fileName = QFileDialog::getOpenFileName(this, "Open file", localDir+"/model_and_data");
    QFile file(fileName);
    data_file_ = QFileInfo(fileName);

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

void LocalView::on_radioButton_binary_clicked()
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

void LocalView::on_radioButton_phaseRecog_clicked()
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

void LocalView::on_radioButton_excitation_clicked()
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

void LocalView::on_radioButton_mnist_clicked()
{
    npzfile_ = "mnist_cav.npz";
    model_name_ = npzfile_.mid(0, npzfile_.indexOf("_"));
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

    if(ui->checkBox->isChecked()){
        ui->checkBox->setChecked(0);  // 取消选中
    }
}

void LocalView::on_slider_unit_sliderMoved(int pos)
{
    ui->spinBox_unit->setValue(pos);
    //    qDebug() << robustness_unit_;
}

void LocalView::on_slider_exptnum_sliderMoved(int pos)
{
    experiment_number_ = pos;
    ui->spinBox_exptnum->setValue(pos);
}

void LocalView::on_spinBox_unit_valueChanged(int pos)
{
    ui->slider_unit->setValue(pos);
    //    qDebug() << robustness_unit_;
}

void LocalView::on_spinBox_exptnum_valueChanged(int pos)
{
    experiment_number_ = pos;
    ui->slider_exptnum->setValue(pos);
}

void LocalView::run_localVeri()
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
    update();

    close_circuit_diagram();
    delete_all_adversary_examples();

    robustness_unit_ = pow(0.1, ui->slider_unit->value());
    experiment_number_ = ui->slider_exptnum->value();

    QString cmd = "python";
    QString excuteFile = "batch_check.py";
    QString unit = QString::number(robustness_unit_);
    QString exptnum = QString::number(experiment_number_);
    QString paramsList;
    QStringList args;
    // qDebug() << exptnum;

    if(npzfile_ != "")
    {
        if(npzfile_.contains("mnist"))
        {
            QString digits = comboBox_digits->current_select_items().join("");
            QString qasmfile = QString("./model_and_data/mnist%1.qasm").arg(digits);
            QString datafile = QString("./model_and_data/mnist%1_data.npz").arg(digits);
            model_name_ = QString("mnist%1").arg(digits);
            args << excuteFile << qasmfile << datafile << unit << exptnum << state_type_;
        }
        else
        {
            QString npzfile = "./model_and_data/" + npzfile_; // npzfile is complete path
            args << excuteFile << npzfile << unit << exptnum << state_type_;
        }
    }
    else    // has imported a file
    {
        QString qasmfile = model_file_.filePath();
        QString datafile = data_file_.filePath();
        args << excuteFile << qasmfile << datafile << unit << exptnum << state_type_;
    }

    // model_name be like: "binary"
    if(model_name_.contains("mnist"))
    {
        if(state_type_ == "pure"){
            ui->checkBox->setChecked(1);
            args << "true";
        }
        else{
            ui->checkBox->setChecked(0);
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
    else
    {
        args << noise_type_ << QString::number(noise_prob_);
    }

    paramsList = cmd + " " + args.join(" ");
    qDebug() << paramsList;

    show_result_tables();

    process = new QProcess(this);
    process->setReadChannel(QProcess::StandardOutput);
    connect(process, SIGNAL(stateChanged(QProcess::ProcessState)), SLOT(stateChanged(QProcess::ProcessState)));
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

    QString error = process->readAllStandardError(); //命令行执行出错的提示
    if(!error.isEmpty()){
        qDebug()<< "Error executing script： " << error; //打印出错提示
    }
}

void LocalView::stateChanged(QProcess::ProcessState state)
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
    bool is_case = circuit_diagram_map.find(model_name_) != circuit_diagram_map.end();
    while (process->bytesAvailable() > 0){
        output_line_ = process->readLine();
        // qDebug() << output_line_;
        output_.append(output_line_);
        ui->textBrowser_output->append(output_line_.simplified());

        if(is_case && output_line_.contains("Starting") && !showed_pdf && !model_name_.contains("mnist"))
        {
            show_circuit_diagram_pdf(localDir+"/Figures/"+circuit_diagram_map[model_name_]);
        }
        else if(!is_case && output_line_.contains(".svg saved") && !showed_pdf)
        {
            show_circuit_diagram_svg(localDir+"/Figures/"+output_line_.mid(0, output_line_.indexOf(".svg saved")+4));
            // show_circuit_diagram_svg(localDir+QString("/Figures/%1_with_%2_%3_model.svg")
            //                                          .arg(model_name_, QString::number(noise_prob_), noise_type_));
        }
        // Verification over, show results and adversary examples.
        else if(output_line_.contains(".csv saved successfully!"))
        {
            csvfile_ = output_line_.mid(0, output_line_.indexOf(".csv saved")+4);
            file_name_ = csvfile_.mid(0, csvfile_.indexOf(".csv"));
            csvfile_ = localDir + "/results/result_tables/" + csvfile_;

            get_table_data("run");

            if(ui->checkBox->isChecked()){
                show_adversary_examples();
            }
            break;
        }
    }
}

void LocalView::show_adversary_examples()
{
    QString img_file = localDir + "/adversary_examples";
    // qDebug() << img_file;

    QDir dir(img_file);
    QStringList mImgNames;

    if (!dir.exists()) mImgNames = QStringList("");

    dir.setFilter(QDir::Files);
    dir.setSorting(QDir::Name);

    mImgNames = dir.entryList();
    // qDebug() << dir.entryList();
    QString digits = comboBox_digits->current_select_items().join("");
    digits = "advExample_" + digits;
    qDebug() << "The selected numbers are: " << digits;
    for (int i = 0; i < mImgNames.size(); ++i)
    {
        if(mImgNames[i].startsWith(digits))
        {
            QLabel *imageLabel_ad;
            imageLabel_ad= new QLabel(ui->scrollAreaWidgetContents_ad);
            imageLabel_ad->setObjectName(QString::fromUtf8("imageLabel_ad_")+QString::number(i));
            imageLabel_ad->setAlignment(Qt::AlignCenter);

            QImage image(img_file + "/" + mImgNames[i]);
            QPixmap pixmap = QPixmap::fromImage(image);
            imageLabel_ad->setPixmap(pixmap);

            ui->verticalLayout_4->addWidget(imageLabel_ad);
            adv_examples.append(mImgNames[i]);
        }
    }
    showed_adexample = true;
}

void LocalView::delete_all_adversary_examples()
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
    adv_examples.clear();
    qDebug() << "delete all adversary examples!";
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

void LocalView::close_circuit_diagram_svg()
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

void LocalView::show_circuit_diagram_svg(QString filename)
{
    qDebug() << "show " << filename;
    QSpacerItem *verticalSpacer1 = new QSpacerItem(20, 40, QSizePolicy::Expanding, QSizePolicy::Expanding);
    ui->verticalLayout_circ->addItem(verticalSpacer1);

    qDebug() << "img_file: " << filename;
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

    QSpacerItem *verticalSpacer2 = new QSpacerItem(20, 40, QSizePolicy::Expanding, QSizePolicy::Expanding);
    ui->verticalLayout_circ->addItem(verticalSpacer2);
    int diff = double(svg_h*2)/double(iris_h) * 1000;
    // qDebug() << diff;
    ui->verticalLayout_circ->insertWidget(1, svgWidget, diff);
    ui->verticalLayout_circ->setStretch(0, 1.5*1000);
    ui->verticalLayout_circ->setStretch(2, 1.5*1000);
    showed_svg = true;
}

void LocalView::show_circuit_diagram_pdf(QString filename)
{
    qDebug() << "img_file: " << filename;
    pdfView = new PdfView(ui->scrollArea_circ);
    pdfView->loadDocument(filename);
    pdfView->setObjectName("pdfView_circ");
    ui->verticalLayout_circ->addWidget(pdfView);
    showed_pdf = true;
}

void LocalView::show_result_tables(){
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
    accuracy_model->setVerticalHeaderLabels(QStringList() << "Rough Verification" << "Accurate Verification");
    times_model->setVerticalHeaderLabels(QStringList() << "Rough Verification" << "Accurate Verification");

    item = new QStandardItem(QString("Rough Verification"));
    //    item->setData(QColor(Qt::gray), Qt::BackgroundRole);
    //    item->setData(QColor(Qt::white), Qt::FontRole);
    item->setTextAlignment(Qt::AlignCenter);
    accuracy_model->setVerticalHeaderItem(0, item);
    item = new QStandardItem(QString("Rough Verification"));
    item->setTextAlignment(Qt::AlignCenter);
    times_model->setVerticalHeaderItem(0, item);

    item = new QStandardItem(QString("Accurate Verification"));
    item->setTextAlignment(Qt::AlignCenter);
    accuracy_model->setVerticalHeaderItem(1, item);
    item = new QStandardItem(QString("Accurate Verification"));
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
void LocalView::get_table_data(QString op){
    // 程序异常结束
    if(op == "run" && process->exitStatus() != QProcess::NormalExit)
    {
        qDebug() << process->exitStatus();
        QMessageBox::warning(this, "Warning", "Program abort.");
        return;
    }

    // 程序正常结束
    QFile file(csvfile_);
    qDebug() << "csvfile: " << csvfile_;

    if(!file.open(QIODevice::ReadOnly | QIODevice::Text)){
        // QMessageBox::warning(this, "Warning", "Unable to open the .csv file: " + csvfile_ + "\n" + file.errorString());
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

void LocalView::on_checkBox_clicked(int state)
{
    if(state == Qt::Checked || state == Qt::PartiallyChecked){
        if(model_name_.contains("mnist") && state_type_ == "pure"){
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

LocalView::~LocalView()
{
    delete ui;
}



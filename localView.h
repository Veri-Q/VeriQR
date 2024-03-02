#ifndef LOCALVIEW_H
#define LOCALVIEW_H

#include <QProcess>
#include <QFileInfo>
#include <QWidget>
#include <QStandardItemModel>
#include <QStandardItem>
#include <QWebEngineView>
// #include "dragCircuit.h"

#include "ui_localview.h"
#include "svgWidget.h"
#include "pdfView.h"
#include "multiSelectComboBox.h"


namespace Ui {
class localView;
}

class LocalView : public QWidget
{
    Q_OBJECT

public:
    Ui::localView *ui;

    explicit LocalView(QWidget *parent = nullptr);
    ~LocalView();

    void init();
    void resizeEvent(QResizeEvent *) override;
    void openFile();
    void model_change_to_ui();
    void saveFile();
    void saveasFile();
    void show_result_tables();
    void show_adversary_examples();
    void delete_all_adversary_examples();
    void show_circuit_diagram_svg(QString filename);
    void show_circuit_diagram_pdf(QString filename);
    void close_circuit_diagram();
    void close_circuit_diagram_svg();
    void close_circuit_diagram_pdf();

public slots:
    void importModel();
    void on_radioButton_importfile_clicked();
    void importData();
    void on_radioButton_binary_clicked();
    void on_radioButton_phaseRecog_clicked();
    void on_radioButton_excitation_clicked();
    void on_radioButton_mnist_clicked();
    void on_radioButton_pure_clicked();
    void on_radioButton_mixed_clicked();
    void on_checkBox_clicked(int state);
    void on_slider_unit_sliderMoved(int position);
    void on_slider_exptnum_sliderMoved(int position);
    void on_spinBox_unit_valueChanged(int arg1);
    void on_spinBox_exptnum_valueChanged(int arg1);
    void run_localVeri();
    void stopProcess();
    void on_read_output();
    void get_table_data(QString op);
    void stateChanged(QProcess::ProcessState state);
    void on_radioButton_bitflip_clicked();
    void on_radioButton_depolarizing_clicked();
    void on_radioButton_phaseflip_clicked();
    void on_radioButton_mixednoise_clicked();
    void on_radioButton_importops_clicked();
    void on_slider_prob_sliderMoved(int);
    void on_doubleSpinBox_prob_valueChanged(double);

private:
    PdfView *pdfView;

    SvgWidget *svgWidget;

    MultiSelectComboBox *comboBox_digits;
    MultiSelectComboBox *comboBox_mixednoise;

    QProcess *process;

    QStandardItemModel *accuracy_model;
    QStandardItemModel *times_model;
    QStandardItem *item;

    QString localDir;
    QFileInfo model_file_;  // 当前选择的qasm模型文件
    QFileInfo data_file_;  // 当前选择的npz数据文件
    QString file_name_;   // txt和csv结果文件命名格式: binary_0.001_3_mixed
    QString model_name_;  // binary
    QString csvfile_;     // ../VeriQR/py_module/Local/results/result_tables/binary_0.001_3_mixed.csv
    QString npzfile_;     // binary_cav.npz

    QString state_type_ = "mixed";
    double robustness_unit_ = 1e-5;
    int experiment_number_ = 5;
    bool need_adversary_examples_ = false;
    QString output_;
    QString output_line_;

    QString noise_types[4] = {"bit_flip", "depolarizing", "phase_flip", "mixed"};
    QStringList mixed_noises_;
    QString noise_type_ = "";
    double noise_prob_ = 0.0;
    QFileInfo kraus_file_;  // 当前选择的kraus operators file

    bool showed_svg = false;
    bool showed_pdf = false;
    bool showed_adexample = false;
    QStringList adv_examples;

    std::map<QString, QString> circuit_diagram_map = {
        {"binary", "binary_model.pdf"},
        {"excitation", "excitation_model.pdf"},
        // {"mnist", "mnist_model.pdf"},
        {"phaseRecog", "phaseRecog.pdf"},
    };

    std::map<QString, QString> noise_name_map = {
        {"BitFlip", "bit_flip"},
        {"Depolarizing", "depolarizing"},
        {"PhaseFlip", "phase_flip"},
    };
};

#endif // LOCALVIEW_H

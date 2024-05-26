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
    void saveFile();
    void saveasFile();
    void show_adversarial_examples();
    void delete_all_adversarial_examples();
    void show_circuit_diagram_svg(QString filename);
    void show_circuit_diagram_pdf();
    void close_circuit_diagram();
    void close_circuit_diagram_svg();
    void close_circuit_diagram_pdf();
    void setItem(QStandardItem *newItem);
    void show_result_tables();
    void get_table_data(QString op);
    void clear_output();
    void reset_all();

public slots:
    void importModel();
    void on_radioButton_importfile_clicked();
    void importData();
    void on_radioButton_qubit_clicked();
    void on_radioButton_phaseRecog_clicked();
    void on_radioButton_excitation_clicked();
    void on_radioButton_mnist_clicked();
    void on_radioButton_pure_clicked();
    void on_radioButton_mixed_clicked();
    void on_checkBox_show_AE_stateChanged(int state);
    void on_checkBox_get_newdata_stateChanged(int state);
    void on_slider_unit_sliderMoved(int position);
    void on_slider_batchnum_sliderMoved(int position);
    void on_spinBox_unit_valueChanged(int arg1);
    void on_spinBox_batchnum_valueChanged(int arg1);
    void on_process_stateChanged(QProcess::ProcessState state);
    void on_radioButton_bitflip_clicked();
    void on_radioButton_depolarizing_clicked();
    void on_radioButton_phaseflip_clicked();
    void on_radioButton_mixednoise_clicked();
    void on_radioButton_importkraus_clicked();
    void on_slider_prob_sliderMoved(int);
    void on_doubleSpinBox_prob_valueChanged(double);
    void run_localVeri();
    void on_read_output();
    void stopProcess();

private:
    QString localDir;

    /* Variables about setting parameters */
    QString npzfile_;       // qubit.npz
    QFileInfo model_file_;  // The currently selected qasm file
    QFileInfo data_file_;   // The currently selected npz file, containing the model and dataset
    QString model_name_;    // e.g. mnist
    QString filename_;  // Used to specify the name of the resulting file for the current model
                        // e.g. qubit_0.001×5_mixed, fashion8_0.001×5_pure_Depolarizing_0.001
    QStringList case_list_ = QStringList(QStringList()<< "qubit"<< "excitation"<< "phaseRecog");
    MultiSelectComboBox *comboBox_digits;

    QString state_type_ = "mixed";
    bool need_to_visualize_AE_ = false;
    bool need_new_dataset_ = false;
    double robustness_unit_ = 1e-5;
    int bacth_num_ = 5;

    // QString noise_types[4] = {"bit_flip", "depolarizing", "phase_flip", "mixed"};
    QString noise_type_;
    double noise_prob_;
    std::map<QString, QString> noise_name_map = {
                                                 {"BitFlip", "bit_flip"},
                                                 {"Depolarizing", "depolarizing"},
                                                 {"PhaseFlip", "phase_flip"},
                                                 };
    QStringList mixed_noises_;
    MultiSelectComboBox *comboBox_mixednoise;
    QFileInfo kraus_file_;  // The currently selected Kraus operators file

    /* Variables about verification program */
    QProcess *process;

    /* Variables about visualization */
    QString output_;
    QString output_line_;

    QString csvfile_;     // e.g. localDir/results/result_tables/mnist_0.001×5_pure_Depolarizing_0.001.csv
    QStandardItemModel *res_model;
    // QStandardItem *item;

    PdfView *pdfView;
    // SvgWidget *svgWidget;
    // Used to indicate whether the image was displayed
    bool showed_svg = false;
    bool showed_pdf = false;
    bool showed_AE_ = false;
    QStringList adv_examples_;
};

#endif // LOCALVIEW_H

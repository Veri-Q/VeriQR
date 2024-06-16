#ifndef GLOBALVIEW_H
#define GLOBALVIEW_H

#include <QProcess>
#include <QFileInfo>
#include <QWidget>
#include "ui_globalview.h"
#include "svgWidget.h"
#include "multiSelectComboBox.h"

namespace Ui {
class globalView;
}

class GlobalView : public QWidget
{
    Q_OBJECT

public:
    explicit GlobalView(QWidget *parent = nullptr);
    ~GlobalView();

    void init();
    void resizeEvent(QResizeEvent *) override;
    void openFile();
    void saveFile();
    void saveasFile();
    bool findFile(QString fileName);
    void show_result_table();
    void insert_data_to_table(int row_index, QString circuit, QString perturbations,
                              QString K, QString VT, QString robust);
    void show_saved_output(QString fileName);
    void show_circuit_diagram(QString img_file);
    void close_circuit_diagram();
    void clear_output();
    void reset_all();
    void exec_calculation(QString cmd, QStringList args);

public slots:
    void on_radioButton_cr_clicked();
    void on_radioButton_aci_clicked();
    void on_radioButton_fct_clicked();
    void on_radioButton_importfile_clicked();
    void importModel();
    void on_radioButton_bitflip_clicked();
    void on_radioButton_depolarizing_clicked();
    void on_radioButton_phaseflip_clicked();
    void on_radioButton_mixednoise_clicked();
    void on_radioButton_importkraus_clicked();
    void on_slider_prob_sliderMoved(int pos);
    void on_doubleSpinBox_prob_valueChanged(double pos);
    void run_calculate_k();
    void stateChanged(QProcess::ProcessState state);
    void on_read_from_terminal_calc();
    void stopProcess();
    void on_slider_epsilon_sliderMoved(int pos);
    void on_doubleSpinBox_epsilon_valueChanged(double pos);
    void on_slider_delta_sliderMoved(int pos);
    void on_doubleSpinBox_delta_valueChanged(double pos);
    void run_globalVeri();
    void on_read_from_terminal_verif();

private:
    Ui::globalView *ui;
    QString globalDir;

    /* Variables about setting parameters */
    QFileInfo model_file_;  // The currently selected qasm file
    QString model_name_;
    QString file_name_;  // Used to specify the name of the resulting file for the current model
                         // e.g. gc_phase_flip_0.0001

    QString noise_types[4] = {"bit_flip", "depolarizing", "phase_flip", "mixed"};
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

    double epsilon_ = 0.0;
    double delta_ = 0.0;

    /* Variables about verification program */
    QProcess* process_cal;
    QProcess* process_veri;
    QString pyfile_ = "global_verif.py";
    int res_count = 0;

    /* Variables about visualization */
    QString output_;
    QString output_line_;

    double lipschitz_;
    double verif_time_;
    bool showed_svg = false;

    QString csvfile_;     // e.g. globalDir/results/result_tables/aci.csv
    QStandardItemModel *res_model;
};

#endif // GLOBALVIEW_H

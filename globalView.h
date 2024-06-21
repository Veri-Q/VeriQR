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
    void openCsvfile();
    void saveTableToCsvfile();
    void saveTableAsFile();
    bool fileExists(QString fileName);
    void showResultTable();
    void insertLineDataToTable(int row_index, QString circuit, QString perturbations,
                              QString K, QString VT, QString if_robust);
    void insertDataToTable(int row_index, int col_index, QString data);
    void showResultFromCsvfile(QString fileName);
    void showOutputFromTxtfile(QString fileName);
    void readLipschitzFromCsvfile(QString fileName);
    void readLipschitzFromTable();
    bool getVerifResultFromCsvfile(QString fileName, double epsilon, double delta);
    void showCircuitDiagram(QString img_file);
    void closeCircuitDiagram();
    void clearOutput();
    void resetAll();
    void execCalculation(QString cmd, QStringList args);
    void importModel();

public slots:
    void on_radioButton_cr_clicked();
    void on_radioButton_aci_clicked();
    void on_radioButton_fct_clicked();
    void on_radioButton_importfile_clicked();
    void on_radioButton_bitflip_clicked();
    void on_radioButton_depolarizing_clicked();
    void on_radioButton_phaseflip_clicked();
    void on_radioButton_mixednoise_clicked();
    void on_radioButton_importkraus_clicked();
    void on_slider_prob_sliderMoved(int pos);
    void on_doubleSpinBox_prob_valueChanged(double pos);
    void run_calculate_k();
    void computationStateChanged(QProcess::ProcessState state);
    void verificationStateChanged(QProcess::ProcessState state);
    void on_read_from_terminal_calc();
    void stop_process();
    void on_slider_epsilon_sliderMoved(int pos);
    void on_doubleSpinBox_epsilon_valueChanged(double pos);
    void on_slider_delta_sliderMoved(int pos);
    void on_doubleSpinBox_delta_valueChanged(double pos);
    void global_verify(double epsilon, double delta, QList<double> kList);
    void run_globalVerify();
    void on_read_from_terminal_verif();

private:
    Ui::globalView *ui;
    QString globalDir;

    /* Variables about setting parameters */
    QFileInfo model_file_;  // The currently selected qasm file
    QString model_name_;
    QString file_name_;  // Used to specify the name of the resulting file for the current model
                         // e.g. cr_PhaseFlip_0.0001

    QString noise_type_ = "bit_flip";
    double noise_prob_;
    QString noise_types_[4] = {"bit_flip", "depolarizing", "phase_flip", "mixed"};
    QMap<QString, QString> noise_name_map_1 = {
                                                 {"BitFlip", "bit_flip"},
                                                 {"Depolarizing", "depolarizing"},
                                                 {"PhaseFlip", "phase_flip"},
                                                 };
    QMap<QString, QString> noise_name_map_2 = {
                                                  {"bit_flip", "BitFlip"},
                                                  {"depolarizing", "Depolarizing"},
                                                  {"phase_flip", "PhaseFlip"},
                                                  };
    QStringList mixed_noises_;
    MultiSelectComboBox *comboBox_mixednoise_;
    QFileInfo kraus_file_;  // The currently selected Kraus operators file

    double epsilon_ = 0.0;
    double delta_ = 0.0;

    /* Variables about verification program */
    QProcess* process_cal_;
    QProcess* process_veri_;
    QString pyfile_ = "global_verif.py";
    QString result_dir_; // e.g. globalDir/results/cr/cr_BitFlip_0.001
    QString csvfile_; // e.g. globalDir/results/cr/cr_BitFlip_0.001/cr_BitFlip_0.001.csv
    QString txtfile_; // e.g. globalDir/results/cr/cr_BitFlip_0.001/cr_BitFlip_0.001.txt

    /* Variables about visualization */
    QString output_;
    int calc_count_ = 0;
    int verif_count_ = 0;
    QStandardItemModel *res_model_;
    double origin_lipschitz_;
    double origin_VT_;
    QString origin_robust_;
    double random_lipschitz_;
    double random_VT_;
    QString random_robust_;
    double specified_lipschitz_;
    double specified_VT_;
    QString specified_robust_;
    bool showed_svg_ = false;
    bool got_result_ = false;
};

#endif // GLOBALVIEW_H

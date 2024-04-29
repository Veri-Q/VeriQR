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
    void openFile();
    void saveFile();
    void saveasFile();
    bool findFile(QString filename);
    void show_saved_results(QString fileName);
    // void show_loss_and_acc_plot();
    void show_circuit_diagram();
    void delete_circuit_diagram();
    void model_change_to_ui();
    void clear_all_information();
    void resizeEvent(QResizeEvent *) override;
    void exec_calculation(QString cmd, QStringList args);

public slots:
    void on_radioButton_importfile_clicked();
    void on_radioButton_cr_clicked();
    void on_radioButton_aci_clicked();
    void on_radioButton_fct_clicked();
    void importModel();
    void on_radioButton_phaseflip_clicked();
    void on_radioButton_bitflip_clicked();
    void on_radioButton_depolarize_clicked();
    void on_radioButton_mixednoise_clicked();
    void on_radioButton_importkraus_clicked();
    void on_slider_prob_sliderMoved(int pos);
    void on_doubleSpinBox_prob_valueChanged(double pos);
    void run_calculate_k();
    void stateChanged(QProcess::ProcessState state);
    void stopProcess();
    void on_read_from_terminal_cal();
    void on_slider_epsilon_sliderMoved(int pos);
    void on_doubleSpinBox_epsilon_valueChanged(double pos);
    void on_slider_delta_sliderMoved(int pos);
    void on_doubleSpinBox_delta_valueChanged(double pos);
    void run_globalVeri();
    void on_read_from_terminal_veri();

private:
    Ui::globalView *ui;

    QProcess* process_cal;
    QProcess* process_veri;

    QString globalDir;
    QString pyfile_ = "qlipschitz.py";

    QFileInfo model_file_; // 当前选择的qasm模型文件
    QString file_name_;    // txt结果文件命名: dice_phase_flip_0.0001
    QString model_name_;

    QString noise_types[4] = {"bit_flip", "depolarizing", "phase_flip", "mixed"};
    QString noise_type_;
    double noise_prob_;
    QStringList mixed_noises_;
    QFileInfo kraus_file_;  // 当前选择的kraus operators file

    std::map<QString, QString> noise_name_map = {
        {"BitFlip", "bit_flip"},
        {"Depolarizing", "depolarizing"},
        {"PhaseFlip", "phase_flip"},
    };

    double lipschitz_;
    double veri_time_;
    double epsilon_ = 0.0;
    double delta_ = 0.0;

    QString output_;
    QString output_line_;

    bool showed_svg = false;

    SvgWidget *svgWidget;
    QSvgRenderer *svgRender;
    MultiSelectComboBox *comboBox_mixednoise;
};

#endif // GLOBALVIEW_H

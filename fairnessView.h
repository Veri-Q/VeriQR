#ifndef FAIRNESSVIEW_H
#define FAIRNESSVIEW_H

#include <QProcess>
#include <QFileInfo>
#include <QWidget>
#include "ui_fairnessview.h"
#include "svgWidget.h"

namespace Ui {
class fairnessView;
}

class FairnessView : public QWidget
{
    Q_OBJECT

public:
    explicit FairnessView(QWidget *parent = nullptr);
    ~FairnessView();

    void init();
    void openFile();
    void saveFile();
    void saveasFile();
    bool findFile(QString filename);
    void show_saved_results(QString fileName);
    void show_loss_and_acc_plot();
    void show_circuit_diagram();
    void delete_circuit_diagram();
    void model_change_to_ui();
    void clear_all_information();
    void resizeEvent(QResizeEvent *) override;
    void exec_process(QString cmd, QStringList args);

public slots:
    void on_radioButton_importfile_clicked();
    void on_radioButton_gc_clicked();
    void on_radioButton_dice_clicked();
    void importModel();  // TODO
    void on_radioButton_phaseflip_clicked();
    void on_radioButton_bitflip_clicked();
    void on_radioButton_depolarize_clicked();
    void on_radioButton_mixed_clicked();
    void on_slider_prob_sliderMoved(int pos);
    void on_doubleSpinBox_prob_valueChanged(double pos);
    void run_fairnessVeri();
    void stateChanged(QProcess::ProcessState state);
    void stopProcess();
    void on_read_from_terminal();

private:
    Ui::fairnessView *ui;

    QProcess* process;

    QString fairDir;

    QString noise_types[4] = {"phase_flip", "bit_flip", "depolarize", "mixed"};
    QString pyfiles[4] = {
        "evaluate_finance_model_gc",
        "evaluate_finance_model_dice",
        "evaluate_trained_model_gc",
        "evaluate_qcnn_model"
    };

    QFileInfo model_file_; // 当前选择的qasm模型文件
    QString file_name_;   // txt结果文件命名: dice_phase_flip_0.0001

    QString noise_type_ = noise_types[3];
    double noise_prob_ = 0.0;

    QString pyfile_;  // evaluate_finance_model_gc

    double lipschitz_;
    double veri_time_;

    QString output_;
    QString output_line_;

    bool showed_loss = false;
    bool showed_svg = false;

    SvgWidget *svgWidget;
    QSvgRenderer *svgRender;
};

#endif // FAIRNESSVIEW_H

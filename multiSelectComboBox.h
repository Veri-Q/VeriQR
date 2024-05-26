#ifndef MULTISELECTCOMBOBOX_H
#define MULTISELECTCOMBOBOX_H


#pragma once

#include <QComboBox>
#include <QListWidget>
#include <QLineEdit>
#include <QCheckBox>
#include <QEvent>

class MultiSelectComboBox : public QComboBox
{
    Q_OBJECT

public:
    MultiSelectComboBox(QWidget *parent = Q_NULLPTR);
    ~MultiSelectComboBox();

    int count() const;  // 返回当前选项条数
    void setMaxSelectNum(int n);  // 限制最大选中选项数量
    QStringList current_select_items();  // 返回当前选中选项

    void addItem(const QString& _text, const QVariant& _variant = QVariant());
    void addItems_for_mnist(const QStringList& _text_list);
    void addItems_for_noise(const QStringList& _text_list);

    // void SetPlaceHolderText(const QString& _text);  // 设置文本框默认文字
    void ResetSelection();  // 下拉框状态恢复默认
    void clear();  // 清空所有内容
    void TextClear(); // 文本框内容清空

    void setCurrentText(const QString& _text); // 设置选中文本--单
    void setCurrentText(const QStringList& _text_list);  // 设置选中文本--多


protected:
    virtual bool eventFilter(QObject *watched,QEvent *event); // 事件过滤器
    virtual void wheelEvent(QWheelEvent *event); //滚轮事件
    virtual void keyPressEvent(QKeyEvent *event); //按键事件

private slots:
    void stateChange(int state);  // 文本框文本变化
    void stateChange_0(int state);
    void stateChange_1(int state);
    void stateChange_2(int state);
    void stateChange_3(int state);
    void stateChange_4(int state);
    void stateChange_5(int state);
    void stateChange_6(int state);
    void stateChange_7(int state);
    void stateChange_8(int state);
    void stateChange_9(int state);
    void stateChange_bitflip(int state);
    void stateChange_depolarizing(int state);
    void stateChange_phaseflip(int state);

signals:
    // 发送当前选中选项
    void selectionChange(const QString _data);

public:
    QLineEdit* line_edit_;
    QListWidget* list_widget_;
    QCheckBox* checkbox_0;
    QCheckBox* checkbox_1;
    QCheckBox* checkbox_2;
    QCheckBox* checkbox_3;
    QCheckBox* checkbox_4;
    QCheckBox* checkbox_5;
    QCheckBox* checkbox_6;
    QCheckBox* checkbox_7;
    QCheckBox* checkbox_8;
    QCheckBox* checkbox_9;

    int max_select_num_ = 0;
    int select_items_count_ = 0;
};


#endif // MULTISELECTCOMBOBOX_H

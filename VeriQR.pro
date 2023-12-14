QT += core gui
QT += svgwidgets
QT += webenginewidgets webchannel network
QT += core5compat
QT += pdf pdfwidgets

CONFIG += use_gold_linker


greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    fairnessView.cpp \
    main.cpp \
    mainwindow.cpp \
    pdfView.cpp \
    robustnessView.cpp \
    svgWidget.cpp

HEADERS += \
    fairnessView.h \
    mainwindow.h \
    pdfView.h \
    robustnessView.h \
    svgWidget.h

FORMS += \
    dragcircuit.ui \
    fairnessview.ui \
    mainwindow.ui \
    robustnessview.ui \

RESOURCES += \
    resources.qrc

DISTFILES += \
    py_module/Robustness/batch_check.py \
    py_module/Robustness/VeriQ.py \
    py_module/Robustness/generate_adversary.py \
    py_module/Robustness/README.md \
    py_module/Fairness/qlipschitz.py \
    py_module/Fairness/evaluate_finance_model_dice.py \
    py_module/Fairness/evaluate_finance_model_gc.py \
    py_module/Fairness/evaluate_qcnn_model.py \
    py_module/Fairness/evaluate_trained_model_gc.py \
    py_module/Fairness/README.md \

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

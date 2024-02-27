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
    globalView.cpp \
    localView.cpp \
    main.cpp \
    mainwindow.cpp \
    multiSelectComboBox.cpp \
    pdfView.cpp \
    svgWidget.cpp

HEADERS += \
    globalView.h \
    localView.h \
    mainwindow.h \
    multiSelectComboBox.h \
    pdfView.h \
    svgWidget.h

FORMS += \
    globalview.ui \
    localview.ui \
    mainwindow.ui

RESOURCES += \
    resources.qrc

DISTFILES += \
    py_module/Local/batch_check.py \
    py_module/Local/VeriQ.py \
    py_module/Local/generate_adversary.py \
    py_module/Local/README.md \
    py_module/Global/qlipschitz.py \
    py_module/Global/evaluate_finance_model_dice.py \
    py_module/Global/evaluate_finance_model_gc.py \
    py_module/Global/evaluate_qcnn_model.py \
    py_module/Global/evaluate_trained_model_gc.py \
    py_module/Global/README.md \

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

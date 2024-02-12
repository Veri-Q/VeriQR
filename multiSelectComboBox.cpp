#include "multiSelectComboBox.h"


MultiSelectComboBox::MultiSelectComboBox(QWidget *parent)
    : QComboBox(parent)
{
    /* 设置文本框 */
    line_edit_ = new QLineEdit();
    line_edit_->setReadOnly(true);
    line_edit_->installEventFilter(this);
    this->setLineEdit(line_edit_);

    list_widget_ = new QListWidget();
    this->setModel(list_widget_->model());
    this->setView(list_widget_);

    // connect(this, static_cast<void (QComboBox::*)(int)>(&QComboBox::activated),
    //         this, &MultiSelectComboBox::itemClicked);
}

void MultiSelectComboBox::stateChange(int state)
{
    QString selected_data = line_edit_->text();
    QString current_check_item;
    // select_dights_count_ = current_select_items().size();

    if(Qt::Checked == state)
    {
        select_items_count_ = 0;
        for(int i = 0; i < list_widget_->count(); i++)
        {
            QCheckBox *check_box = static_cast<QCheckBox*>(list_widget_->itemWidget(list_widget_->item(i)));
            if(check_box->isChecked())
            {
                select_items_count_++;
                if(!selected_data.contains(check_box->text()))
                {
                    selected_data.append(check_box->text()).append(";");
                    current_check_item = check_box->text();
                    qDebug() << "check " << current_check_item;
                    break;
                }
            }
        }
        qDebug() << "select_dights_count_ = " << select_items_count_;
        if(select_items_count_ <= max_select_num_)
        {
            if(!selected_data.isEmpty())
            {
                // selected_data.chop(1);
                line_edit_->setText(selected_data);
            }
            else
            {
                line_edit_->clear();
            }
        }
        else
        {
            for(int i = 0; i < list_widget_->count(); i++)
            {
                QCheckBox *check_box = static_cast<QCheckBox*>(list_widget_->itemWidget(list_widget_->item(i)));
                if(check_box->text() == current_check_item)
                {
                    qDebug() << "not allow to check " << current_check_item;
                    check_box->setChecked(false);
                    select_items_count_--;
                }
            }
        }
    }
    else if(Qt::Unchecked == state)
    {
        select_items_count_ = 0;
        for(int i = 0; i < list_widget_->count(); i++)
        {
            QCheckBox *check_box = static_cast<QCheckBox*>(list_widget_->itemWidget(list_widget_->item(i)));
            if(selected_data.contains(check_box->text()) && !check_box->isChecked())
            {
                int ind = selected_data.indexOf(check_box->text());
                selected_data = selected_data.mid(0, ind) + selected_data.mid(ind+check_box->text().size()+1);
                qDebug() << "Uncheck " << check_box->text() << " and get " << selected_data;
            }
            else if(check_box->isChecked())
            {
                select_items_count_++;
            }
        }
        if(!selected_data.isEmpty())
        {
            line_edit_->setText(selected_data);
        }
        else
        {
            line_edit_->clear();
        }
    }
    qDebug() << "select " << select_items_count_ << " items: " << current_select_items().join(", ");
    qDebug() << "***********************************************";
}

void MultiSelectComboBox::stateChange_0(int state)
{
    QString selected_data = line_edit_->text();
    QCheckBox *check_box = static_cast<QCheckBox*>(list_widget_->itemWidget(list_widget_->item(0)));
    qDebug() << "state " << state;
    if(Qt::Checked == state)
    {
        int s = current_select_items().count();
        qDebug() << "selected " << s;
        if(s+1 <= max_select_num_)
        {
            qDebug() << "selected_data " << selected_data;
            selected_data.append(check_box->text()).append(";");
            qDebug() << "selected_data " << selected_data;
            if(!selected_data.isEmpty())
            {
                line_edit_->setText(selected_data);
                // select_items_count_++;
                qDebug() << "Check " << check_box->text();
            }
            else
            {
                line_edit_->clear();
            }
        }
        else
        {
            check_box->setChecked(false);
        }
    }
    else
    {
        qDebug() << "selected_data " << selected_data;
        if(selected_data.contains(check_box->text()))
        {
            int ind = selected_data.indexOf(check_box->text());
            if(ind < selected_data.size())
            {
                selected_data = selected_data.mid(0, ind);
            }
            if(ind+check_box->text().size()+1 < selected_data.size())
            {
                selected_data += selected_data.mid(ind+check_box->text().size()+1);
            }
        }
        qDebug() << "selected_data " << selected_data;
        if(!selected_data.isEmpty())
        {
            line_edit_->setText(selected_data);
            // select_items_count_--;
            qDebug() << "UnCheck " << check_box->text();
        }
        else
        {
            line_edit_->clear();
        }
    }
    qDebug() << current_select_items();
    qDebug() << current_select_items().count() << " items: " << current_select_items().join(", ");
    qDebug() << "***********************************************";
}

void MultiSelectComboBox::stateChange_1(int state)
{
    QString selected_data = line_edit_->text();
    QCheckBox *check_box = static_cast<QCheckBox*>(list_widget_->itemWidget(list_widget_->item(1)));
    qDebug() << "state " << state;
    if(Qt::Checked == state)
    {
        int s = current_select_items().count();
        qDebug() << "selected " << s;
        if(s+1 <= max_select_num_)
        {
            qDebug() << "selected_data " << selected_data;
            selected_data.append(check_box->text()).append(";");
            qDebug() << "selected_data " << selected_data;
            if(!selected_data.isEmpty())
            {
                line_edit_->setText(selected_data);
                // select_items_count_++;
                qDebug() << "Check " << check_box->text();
            }
            else
            {
                line_edit_->clear();
            }
        }
        else
        {
            check_box->setChecked(false);
        }
    }
    else
    {
        qDebug() << "selected_data " << selected_data;
        if(selected_data.contains(check_box->text()))
        {
            int ind = selected_data.indexOf(check_box->text());
            if(ind < selected_data.size())
            {
                selected_data = selected_data.mid(0, ind);
            }
            if(ind+check_box->text().size()+1 < selected_data.size())
            {
                selected_data += selected_data.mid(ind+check_box->text().size()+1);
            }
        }
        qDebug() << "selected_data " << selected_data;
        if(!selected_data.isEmpty())
        {
            line_edit_->setText(selected_data);
            // select_items_count_--;
            qDebug() << "UnCheck " << check_box->text();
        }
        else
        {
            line_edit_->clear();
        }
    }
    qDebug() << current_select_items().size() << " items: " << current_select_items().join(", ");
    qDebug() << "***********************************************";
}

void MultiSelectComboBox::stateChange_2(int state)
{
    QString selected_data = line_edit_->text();
    QCheckBox *check_box = static_cast<QCheckBox*>(list_widget_->itemWidget(list_widget_->item(2)));
    qDebug() << "state " << state;
    if(Qt::Checked == state)
    {
        int s = current_select_items().count();
        qDebug() << "selected " << s;
        if(s+1 <= max_select_num_)
        {
            qDebug() << "selected_data " << selected_data;
            selected_data.append(check_box->text()).append(";");
            qDebug() << "selected_data " << selected_data;
            if(!selected_data.isEmpty())
            {
                line_edit_->setText(selected_data);
                // select_items_count_++;
                qDebug() << "Check " << check_box->text();
            }
            else
            {
                line_edit_->clear();
            }
        }
        else
        {
            check_box->setChecked(false);
        }
    }
    else
    {
        qDebug() << "selected_data " << selected_data;
        if(selected_data.contains(check_box->text()))
        {
            int ind = selected_data.indexOf(check_box->text());
            if(ind < selected_data.size())
            {
                selected_data = selected_data.mid(0, ind);
            }
            if(ind+check_box->text().size()+1 < selected_data.size())
            {
                selected_data += selected_data.mid(ind+check_box->text().size()+1);
            }
        }
        qDebug() << "selected_data " << selected_data;
        if(!selected_data.isEmpty())
        {
            line_edit_->setText(selected_data);
            // select_items_count_--;
            qDebug() << "UnCheck " << check_box->text();
        }
        else
        {
            line_edit_->clear();
        }
    }
    qDebug() << current_select_items().size() << " items: " << current_select_items().join(", ");
    qDebug() << "***********************************************";
}

void MultiSelectComboBox::stateChange_3(int state)
{
    QString selected_data = line_edit_->text();
    QCheckBox *check_box = static_cast<QCheckBox*>(list_widget_->itemWidget(list_widget_->item(3)));
    qDebug() << "state " << state;
    if(Qt::Checked == state)
    {
        int s = current_select_items().count();
        qDebug() << "selected " << s;
        if(s+1 <= max_select_num_)
        {
            qDebug() << "selected_data " << selected_data;
            selected_data.append(check_box->text()).append(";");
            qDebug() << "selected_data " << selected_data;
            if(!selected_data.isEmpty())
            {
                line_edit_->setText(selected_data);
                // select_items_count_++;
                qDebug() << "Check " << check_box->text();
            }
            else
            {
                line_edit_->clear();
            }
        }
        else
        {
            check_box->setChecked(false);
        }
    }
    else
    {
        qDebug() << "selected_data " << selected_data;
        if(selected_data.contains(check_box->text()))
        {
            int ind = selected_data.indexOf(check_box->text());
            if(ind < selected_data.size())
            {
                selected_data = selected_data.mid(0, ind);
            }
            if(ind+check_box->text().size()+1 < selected_data.size())
            {
                selected_data += selected_data.mid(ind+check_box->text().size()+1);
            }
        }
        qDebug() << "selected_data " << selected_data;
        if(!selected_data.isEmpty())
        {
            line_edit_->setText(selected_data);
            // select_items_count_--;
            qDebug() << "UnCheck " << check_box->text();
        }
        else
        {
            line_edit_->clear();
        }
    }
    qDebug() << current_select_items().size() << " items: " << current_select_items().join(", ");
    qDebug() << "***********************************************";
}

void MultiSelectComboBox::stateChange_4(int state)
{
    QString selected_data = line_edit_->text();
    QCheckBox *check_box = static_cast<QCheckBox*>(list_widget_->itemWidget(list_widget_->item(4)));
    qDebug() << "state " << state;
    if(Qt::Checked == state)
    {
        int s = current_select_items().count();
        qDebug() << "selected " << s;
        if(s+1 <= max_select_num_)
        {
            qDebug() << "selected_data " << selected_data;
            selected_data.append(check_box->text()).append(";");
            qDebug() << "selected_data " << selected_data;
            if(!selected_data.isEmpty())
            {
                line_edit_->setText(selected_data);
                // select_items_count_++;
                qDebug() << "Check " << check_box->text();
            }
            else
            {
                line_edit_->clear();
            }
        }
        else
        {
            check_box->setChecked(false);
        }
    }
    else
    {
        qDebug() << "selected_data " << selected_data;
        if(selected_data.contains(check_box->text()))
        {
            int ind = selected_data.indexOf(check_box->text());
            if(ind < selected_data.size())
            {
                selected_data = selected_data.mid(0, ind);
            }
            if(ind+check_box->text().size()+1 < selected_data.size())
            {
                selected_data += selected_data.mid(ind+check_box->text().size()+1);
            }
        }
        qDebug() << "selected_data " << selected_data;
        if(!selected_data.isEmpty())
        {
            line_edit_->setText(selected_data);
            // select_items_count_--;
            qDebug() << "UnCheck " << check_box->text();
        }
        else
        {
            line_edit_->clear();
        }
    }
    qDebug() << current_select_items().size() << " items: " << current_select_items().join(", ");
    qDebug() << "***********************************************";
}

void MultiSelectComboBox::stateChange_5(int state)
{
    QString selected_data = line_edit_->text();
    QCheckBox *check_box = static_cast<QCheckBox*>(list_widget_->itemWidget(list_widget_->item(5)));
    qDebug() << "state " << state;
    if(Qt::Checked == state)
    {
        int s = current_select_items().count();
        qDebug() << "selected " << s;
        if(s+1 <= max_select_num_)
        {
            qDebug() << "selected_data " << selected_data;
            selected_data.append(check_box->text()).append(";");
            qDebug() << "selected_data " << selected_data;
            if(!selected_data.isEmpty())
            {
                line_edit_->setText(selected_data);
                // select_items_count_++;
                qDebug() << "Check " << check_box->text();
            }
            else
            {
                line_edit_->clear();
            }
        }
        else
        {
            check_box->setChecked(false);
        }
    }
    else
    {
        qDebug() << "selected_data " << selected_data;
        if(selected_data.contains(check_box->text()))
        {
            int ind = selected_data.indexOf(check_box->text());
            if(ind < selected_data.size())
            {
                selected_data = selected_data.mid(0, ind);
            }
            if(ind+check_box->text().size()+1 < selected_data.size())
            {
                selected_data += selected_data.mid(ind+check_box->text().size()+1);
            }
        }
        qDebug() << "selected_data " << selected_data;
        if(!selected_data.isEmpty())
        {
            line_edit_->setText(selected_data);
            // select_items_count_--;
            qDebug() << "UnCheck " << check_box->text();
        }
        else
        {
            line_edit_->clear();
        }
    }
    qDebug() << current_select_items().size() << " items: " << current_select_items().join(", ");
    qDebug() << "***********************************************";
}

void MultiSelectComboBox::stateChange_6(int state)
{
    QString selected_data = line_edit_->text();
    QCheckBox *check_box = static_cast<QCheckBox*>(list_widget_->itemWidget(list_widget_->item(6)));
    qDebug() << "state " << state;
    if(Qt::Checked == state)
    {
        int s = current_select_items().count();
        qDebug() << "selected " << s;
        if(s+1 <= max_select_num_)
        {
            qDebug() << "selected_data " << selected_data;
            selected_data.append(check_box->text()).append(";");
            qDebug() << "selected_data " << selected_data;
            if(!selected_data.isEmpty())
            {
                line_edit_->setText(selected_data);
                // select_items_count_++;
                qDebug() << "Check " << check_box->text();
            }
            else
            {
                line_edit_->clear();
            }
        }
        else
        {
            check_box->setChecked(false);
        }
    }
    else
    {
        qDebug() << "selected_data " << selected_data;
        if(selected_data.contains(check_box->text()))
        {
            int ind = selected_data.indexOf(check_box->text());
            if(ind < selected_data.size())
            {
                selected_data = selected_data.mid(0, ind);
            }
            if(ind+check_box->text().size()+1 < selected_data.size())
            {
                selected_data += selected_data.mid(ind+check_box->text().size()+1);
            }
        }
        qDebug() << "selected_data " << selected_data;
        if(!selected_data.isEmpty())
        {
            line_edit_->setText(selected_data);
            // select_items_count_--;
            qDebug() << "UnCheck " << check_box->text();
        }
        else
        {
            line_edit_->clear();
        }
    }
    qDebug() << current_select_items().size() << " items: " << current_select_items().join(", ");
    qDebug() << "***********************************************";
}

void MultiSelectComboBox::stateChange_7(int state)
{
    QString selected_data = line_edit_->text();
    QCheckBox *check_box = static_cast<QCheckBox*>(list_widget_->itemWidget(list_widget_->item(7)));
    qDebug() << "state " << state;
    if(Qt::Checked == state)
    {
        int s = current_select_items().count();
        qDebug() << "selected " << s;
        if(s+1 <= max_select_num_)
        {
            qDebug() << "selected_data " << selected_data;
            selected_data.append(check_box->text()).append(";");
            qDebug() << "selected_data " << selected_data;
            if(!selected_data.isEmpty())
            {
                line_edit_->setText(selected_data);
                // select_items_count_++;
                qDebug() << "Check " << check_box->text();
            }
            else
            {
                line_edit_->clear();
            }
        }
        else
        {
            check_box->setChecked(false);
        }
    }
    else
    {
        qDebug() << "selected_data " << selected_data;
        if(selected_data.contains(check_box->text()))
        {
            int ind = selected_data.indexOf(check_box->text());
            if(ind < selected_data.size())
            {
                selected_data = selected_data.mid(0, ind);
            }
            if(ind+check_box->text().size()+1 < selected_data.size())
            {
                selected_data += selected_data.mid(ind+check_box->text().size()+1);
            }
        }
        qDebug() << "selected_data " << selected_data;
        if(!selected_data.isEmpty())
        {
            line_edit_->setText(selected_data);
            // select_items_count_--;
            qDebug() << "UnCheck " << check_box->text();
        }
        else
        {
            line_edit_->clear();
        }
    }
    qDebug() << current_select_items().size() << " items: " << current_select_items().join(", ");
    qDebug() << "***********************************************";
}

void MultiSelectComboBox::stateChange_8(int state)
{
    QString selected_data = line_edit_->text();
    QCheckBox *check_box = static_cast<QCheckBox*>(list_widget_->itemWidget(list_widget_->item(8)));
    qDebug() << "state " << state;
    if(Qt::Checked == state)
    {
        int s = current_select_items().count();
        qDebug() << "selected " << s;
        if(s+1 <= max_select_num_)
        {
            qDebug() << "selected_data " << selected_data;
            selected_data.append(check_box->text()).append(";");
            qDebug() << "selected_data " << selected_data;
            if(!selected_data.isEmpty())
            {
                line_edit_->setText(selected_data);
                // select_items_count_++;
                qDebug() << "Check " << check_box->text();
            }
            else
            {
                line_edit_->clear();
            }
        }
        else
        {
            check_box->setChecked(false);
        }
    }
    else
    {
        qDebug() << "selected_data " << selected_data;
        if(selected_data.contains(check_box->text()))
        {
            int ind = selected_data.indexOf(check_box->text());
            if(ind < selected_data.size())
            {
                selected_data = selected_data.mid(0, ind);
            }
            if(ind+check_box->text().size()+1 < selected_data.size())
            {
                selected_data += selected_data.mid(ind+check_box->text().size()+1);
            }
        }
        qDebug() << "selected_data " << selected_data;
        if(!selected_data.isEmpty())
        {
            line_edit_->setText(selected_data);
            // select_items_count_--;
            qDebug() << "UnCheck " << check_box->text();
        }
        else
        {
            line_edit_->clear();
        }
    }
    qDebug() << current_select_items().size() << " items: " << current_select_items().join(", ");
    qDebug() << "***********************************************";
}

void MultiSelectComboBox::stateChange_9(int state)
{
    QString selected_data = line_edit_->text();
    QCheckBox *check_box = static_cast<QCheckBox*>(list_widget_->itemWidget(list_widget_->item(9)));
    qDebug() << "state " << state;
    if(Qt::Checked == state)
    {
        int s = current_select_items().count();
        qDebug() << "selected " << s;
        if(s+1 <= max_select_num_)
        {
            qDebug() << "selected_data " << selected_data;
            selected_data.append(check_box->text()).append(";");
            qDebug() << "selected_data " << selected_data;
            if(!selected_data.isEmpty())
            {
                line_edit_->setText(selected_data);
                // select_items_count_++;
                qDebug() << "Check " << check_box->text();
            }
            else
            {
                line_edit_->clear();
            }
        }
        else
        {
            check_box->setChecked(false);
        }
    }
    else
    {
        qDebug() << "selected_data " << selected_data;
        if(selected_data.contains(check_box->text()))
        {
            int ind = selected_data.indexOf(check_box->text());
            if(ind < selected_data.size())
            {
                selected_data = selected_data.mid(0, ind);
            }
            if(ind+check_box->text().size()+1 < selected_data.size())
            {
                selected_data += selected_data.mid(ind+check_box->text().size()+1);
            }
        }
        qDebug() << "selected_data " << selected_data;
        if(!selected_data.isEmpty())
        {
            line_edit_->setText(selected_data);
            // select_items_count_--;
            qDebug() << "UnCheck " << check_box->text();
        }
        else
        {
            line_edit_->clear();
        }
    }
    qDebug() << current_select_items().size() << " items: " << current_select_items().join(", ");
    qDebug() << "***********************************************";
}

void MultiSelectComboBox::stateChange_bitflip(int state)
{
    QString selected_data = line_edit_->text();
    QCheckBox *check_box = static_cast<QCheckBox*>(list_widget_->itemWidget(list_widget_->item(0)));
    if(Qt::Checked == state)
    {
        select_items_count_++;
        if(select_items_count_ <= max_select_num_)
        {
            selected_data.append(check_box->text()).append(";");
            if(!selected_data.isEmpty())
            {
                // selected_data.chop(1);
                line_edit_->setText(selected_data);
                qDebug() << "Check " << check_box->text();
            }
            else
            {
                line_edit_->clear();
            }
        }
        else
        {
            check_box->setChecked(false);
            select_items_count_--;
        }
    }
    else if(Qt::Unchecked == state)
    {
        select_items_count_--;

        qDebug() << "selected_data " << selected_data;
        int ind = selected_data.indexOf(check_box->text());
        selected_data = selected_data.mid(0, ind) + selected_data.mid(ind+check_box->text().size()+1);
        qDebug() << "selected_data " << selected_data;
        if(!selected_data.isEmpty())
        {
            qDebug() << "UnCheck " << check_box->text();
            line_edit_->setText(selected_data);
        }
        else
        {
            line_edit_->clear();
        }
    }
    qDebug() << select_items_count_ << " items: " << current_select_items().join(", ");
    qDebug() << "***********************************************";
}

void MultiSelectComboBox::stateChange_depolarizing(int state)
{
    QString selected_data = line_edit_->text();
    QCheckBox *check_box = static_cast<QCheckBox*>(list_widget_->itemWidget(list_widget_->item(1)));
    if(Qt::Checked == state)
    {
        select_items_count_++;
        if(select_items_count_ <= max_select_num_)
        {
            selected_data.append(check_box->text()).append(";");
            if(!selected_data.isEmpty())
            {
                // selected_data.chop(1);
                line_edit_->setText(selected_data);
                qDebug() << "Check " << check_box->text();
            }
            else
            {
                line_edit_->clear();
            }
        }
        else
        {
            check_box->setChecked(false);
            select_items_count_--;
        }
    }
    else if(Qt::Unchecked == state)
    {
        select_items_count_--;

        qDebug() << "selected_data " << selected_data;
        int ind = selected_data.indexOf(check_box->text());
        selected_data = selected_data.mid(0, ind) + selected_data.mid(ind+check_box->text().size()+1);
        qDebug() << "selected_data " << selected_data;
        if(!selected_data.isEmpty())
        {
            qDebug() << "UnCheck " << check_box->text();
            line_edit_->setText(selected_data);
        }
        else
        {
            line_edit_->clear();
        }
    }
    qDebug() << select_items_count_ << " items: " << current_select_items().join(", ");
    qDebug() << "***********************************************";
}

void MultiSelectComboBox::stateChange_phaseflip(int state)
{
    QString selected_data = line_edit_->text();
    QCheckBox *check_box = static_cast<QCheckBox*>(list_widget_->itemWidget(list_widget_->item(2)));
    if(Qt::Checked == state)
    {
        select_items_count_++;
        if(select_items_count_ <= max_select_num_)
        {
            selected_data.append(check_box->text()).append(";");
            if(!selected_data.isEmpty())
            {
                // selected_data.chop(1);
                line_edit_->setText(selected_data);
                qDebug() << "Check " << check_box->text();
            }
            else
            {
                line_edit_->clear();
            }
        }
        else
        {
            check_box->setChecked(false);
            select_items_count_--;
        }
    }
    else if(Qt::Unchecked == state)
    {
        select_items_count_--;

        qDebug() << "selected_data " << selected_data;
        int ind = selected_data.indexOf(check_box->text());
        selected_data = selected_data.mid(0, ind) + selected_data.mid(ind+check_box->text().size()+1);
        qDebug() << "selected_data " << selected_data;
        if(!selected_data.isEmpty())
        {
            qDebug() << "UnCheck " << check_box->text();
            line_edit_->setText(selected_data);
        }
        else
        {
            line_edit_->clear();
        }
    }
    qDebug() << select_items_count_ << " items: " << current_select_items().join(", ");
    qDebug() << "***********************************************";
}

void MultiSelectComboBox::addItem(const QString& _text, const QVariant& _variant)
{
    Q_UNUSED(_variant);

    QListWidgetItem* item_1 = new QListWidgetItem(list_widget_);
    QCheckBox* checkbox_1 = new QCheckBox(this);
    checkbox_1->setText(_text);
    list_widget_->addItem(item_1);
    list_widget_->setItemWidget(item_1, checkbox_1);
    connect(checkbox_1, &QCheckBox::stateChanged, this, &MultiSelectComboBox::stateChange_1);
}

void MultiSelectComboBox::addItems_for_mnist(const QStringList& _text_list)
{
    if(_text_list.size() != 10) return;

    QListWidgetItem* item_0 = new QListWidgetItem(list_widget_);
    checkbox_0 = new QCheckBox(this);
    checkbox_0->setText(_text_list[0]);
    list_widget_->addItem(item_0);
    list_widget_->setItemWidget(item_0, checkbox_0);

    QListWidgetItem* item_1 = new QListWidgetItem(list_widget_);
    checkbox_1 = new QCheckBox(this);
    checkbox_1->setText(_text_list[1]);
    list_widget_->addItem(item_1);
    list_widget_->setItemWidget(item_1, checkbox_1);

    QListWidgetItem* item_2 = new QListWidgetItem(list_widget_);
    checkbox_2 = new QCheckBox(this);
    checkbox_2->setText(_text_list[2]);
    list_widget_->addItem(item_2);
    list_widget_->setItemWidget(item_2, checkbox_2);

    QListWidgetItem* item_3 = new QListWidgetItem(list_widget_);
    checkbox_3 = new QCheckBox(this);
    checkbox_3->setText(_text_list[3]);
    list_widget_->addItem(item_3);
    list_widget_->setItemWidget(item_3, checkbox_3);

    QListWidgetItem* item_4 = new QListWidgetItem(list_widget_);
    checkbox_4 = new QCheckBox(this);
    checkbox_4->setText(_text_list[4]);
    list_widget_->addItem(item_4);
    list_widget_->setItemWidget(item_4, checkbox_4);

    QListWidgetItem* item_5 = new QListWidgetItem(list_widget_);
    checkbox_5 = new QCheckBox(this);
    checkbox_5->setText(_text_list[5]);
    list_widget_->addItem(item_5);
    list_widget_->setItemWidget(item_5, checkbox_5);

    QListWidgetItem* item_6 = new QListWidgetItem(list_widget_);
    checkbox_6 = new QCheckBox(this);
    checkbox_6->setText(_text_list[6]);
    list_widget_->addItem(item_6);
    list_widget_->setItemWidget(item_6, checkbox_6);

    QListWidgetItem* item_7 = new QListWidgetItem(list_widget_);
    checkbox_7 = new QCheckBox(this);
    checkbox_7->setText(_text_list[7]);
    list_widget_->addItem(item_7);
    list_widget_->setItemWidget(item_7, checkbox_7);

    QListWidgetItem* item_8 = new QListWidgetItem(list_widget_);
    checkbox_8 = new QCheckBox(this);
    checkbox_8->setText(_text_list[8]);
    list_widget_->addItem(item_8);
    list_widget_->setItemWidget(item_8, checkbox_8);

    QListWidgetItem* item_9 = new QListWidgetItem(list_widget_);
    checkbox_9 = new QCheckBox(this);
    checkbox_9->setText(_text_list[9]);
    list_widget_->addItem(item_9);
    list_widget_->setItemWidget(item_9, checkbox_9);

    connect(checkbox_0, &QCheckBox::stateChanged, this, &MultiSelectComboBox::stateChange_0);
    connect(checkbox_1, &QCheckBox::stateChanged, this, &MultiSelectComboBox::stateChange_1);
    connect(checkbox_2, &QCheckBox::stateChanged, this, &MultiSelectComboBox::stateChange_2);
    connect(checkbox_3, &QCheckBox::stateChanged, this, &MultiSelectComboBox::stateChange_3);
    connect(checkbox_4, &QCheckBox::stateChanged, this, &MultiSelectComboBox::stateChange_4);
    connect(checkbox_5, &QCheckBox::stateChanged, this, &MultiSelectComboBox::stateChange_5);
    connect(checkbox_6, &QCheckBox::stateChanged, this, &MultiSelectComboBox::stateChange_6);
    connect(checkbox_7, &QCheckBox::stateChanged, this, &MultiSelectComboBox::stateChange_7);
    connect(checkbox_8, &QCheckBox::stateChanged, this, &MultiSelectComboBox::stateChange_8);
    connect(checkbox_9, &QCheckBox::stateChanged, this, &MultiSelectComboBox::stateChange_9);
}

void MultiSelectComboBox::addItems_for_noise(const QStringList& _text_list)
{
    if(_text_list.size() != 3) return;

    QListWidgetItem* item_3 = new QListWidgetItem(list_widget_);
    checkbox_3 = new QCheckBox(this);
    checkbox_3->setText(_text_list[2]);
    list_widget_->addItem(item_3);
    list_widget_->setItemWidget(item_3, checkbox_3);
    connect(checkbox_3, &QCheckBox::stateChanged, this, &MultiSelectComboBox::stateChange_bitflip);

    QListWidgetItem* item_1 = new QListWidgetItem(list_widget_);
    checkbox_1 = new QCheckBox(this);
    checkbox_1->setText(_text_list[0]);
    list_widget_->addItem(item_1);
    list_widget_->setItemWidget(item_1, checkbox_1);
    connect(checkbox_1, &QCheckBox::stateChanged, this, &MultiSelectComboBox::stateChange_depolarizing);

    QListWidgetItem* item_2 = new QListWidgetItem(list_widget_);
    checkbox_2 = new QCheckBox(this);
    checkbox_2->setText(_text_list[1]);
    list_widget_->addItem(item_2);
    list_widget_->setItemWidget(item_2, checkbox_2);
    connect(checkbox_2, &QCheckBox::stateChanged, this, &MultiSelectComboBox::stateChange_phaseflip);
}

void MultiSelectComboBox::setMaxSelectNum(int n)
{
    max_select_num_ = n;
}

QStringList MultiSelectComboBox::current_select_items()
{
    QStringList text_list;
    if (!line_edit_->text().isEmpty())
    {
        text_list = line_edit_->text().split(';'); // 以;为分隔符分割字符串
    }
    if(text_list.contains(""))
    {
        text_list.removeLast();
    }
    return text_list;
}

int MultiSelectComboBox::count() const
{
    return list_widget_->count();
}

// void MultiSelectComboBox::SetPlaceHolderText(const QString& _text)
// {
//     line_edit_->setPlaceholderText(_text);
// }

void MultiSelectComboBox::ResetSelection()
{
    int count = list_widget_->count();
    for (int i = 0; i < count; i++)
    {
        //获取对应位置的QWidget对象
        QWidget *widget = list_widget_->itemWidget(list_widget_->item(i));
        //将QWidget对象转换成对应的类型
        QCheckBox *check_box = static_cast<QCheckBox*>(widget);
        check_box->setChecked(false);
    }
}

void MultiSelectComboBox::clear()
{
    line_edit_->clear();
    list_widget_->clear();
    QListWidgetItem* currentItem = new QListWidgetItem(list_widget_);
    list_widget_->addItem(currentItem);
    select_items_count_ = 0;
}

void MultiSelectComboBox::TextClear()
{
    line_edit_->clear();
    select_items_count_ = 0;
    ResetSelection();
}

void MultiSelectComboBox::setCurrentText(const QString& _text)
{
    int count = list_widget_->count();
    for (int i = 0; i < count; i++)
    {
        //获取对应位置的QWidget对象
        QWidget *widget = list_widget_->itemWidget(list_widget_->item(i));
        //将QWidget对象转换成对应的类型
        QCheckBox *check_box = static_cast<QCheckBox*>(widget);
        if (_text.compare(check_box->text()))
        {
            check_box->setChecked(true);
        }
    }
}

void MultiSelectComboBox::setCurrentText(const QStringList& _text_list)
{
    int count = list_widget_->count();
    for (int i = 0; i < count; i++)
    {
        //获取对应位置的QWidget对象
        QWidget *widget = list_widget_->itemWidget(list_widget_->item(i));
        //将QWidget对象转换成对应的类型
        QCheckBox *check_box = static_cast<QCheckBox*>(widget);
        if (_text_list.contains(check_box->text()))
        {
            check_box->setChecked(true);
        }
    }
}

bool MultiSelectComboBox::eventFilter(QObject *watched, QEvent *event)
{
    // 设置点击输入框也可以弹出下拉框
    if (watched == line_edit_ && event->type() == QEvent::MouseButtonRelease && this->isEnabled())
    {
        showPopup();
        return true;
    }
    return false;
}

void MultiSelectComboBox::wheelEvent(QWheelEvent *event)
{
    // 禁用QComboBox默认的滚轮事件
    Q_UNUSED(event);
}

void MultiSelectComboBox::keyPressEvent(QKeyEvent *event)
{
    QComboBox::keyPressEvent(event);
}

MultiSelectComboBox::~MultiSelectComboBox()
{

}

# Generated by Django 2.2.7 on 2020-03-16 02:37

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('boards', '0002_medical_history_date'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='examination',
            name='cp',
        ),
    ]

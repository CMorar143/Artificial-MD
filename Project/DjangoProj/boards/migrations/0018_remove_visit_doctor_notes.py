# Generated by Django 2.2.7 on 2020-03-08 19:50

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('boards', '0017_auto_20200308_1240'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='visit',
            name='doctor_notes',
        ),
    ]

# Generated by Django 2.2.7 on 2020-03-05 03:18

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('boards', '0014_auto_20200305_0156'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='examination',
            name='pulse_type',
        ),
    ]
# Generated by Django 2.2.7 on 2020-03-27 23:58

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('boards', '0010_auto_20200327_2356'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='examination',
            name='protein',
        ),
    ]

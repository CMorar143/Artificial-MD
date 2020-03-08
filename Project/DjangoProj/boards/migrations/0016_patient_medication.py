# Generated by Django 2.2.7 on 2020-03-08 12:26

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('boards', '0015_remove_examination_pulse_type'),
    ]

    operations = [
        migrations.CreateModel(
            name='Patient_Medication',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('medication', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='boards.Medication')),
                ('patient', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='boards.Patient')),
            ],
        ),
    ]